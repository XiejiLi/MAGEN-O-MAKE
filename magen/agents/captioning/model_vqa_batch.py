import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from collections import defaultdict

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]


class LLaVADataset(Dataset):
    """Dataset class for LLaVA batch inference without padding"""
    
    def __init__(self, questions, image_folder, image_processor, model_config, tokenizer, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.conv_mode = conv_mode
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        line = self.questions[idx]
        question_id = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        
        # Store original prompt for output
        cur_prompt = qs
        
        # Process question text with image tokens
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # Create conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize the prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Load and process image
        try:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            # Exit on image processing error to maintain data integrity
            exit(1)
        
        return {
            'question_id': question_id,
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'image_size': image.size,
            'prompt': cur_prompt,
            'seq_len': len(input_ids)  # Store sequence length for grouping
        }


def group_by_exact_length(dataset):
    """Group samples by exact sequence length to avoid padding completely"""
    length_groups = defaultdict(list)
    
    # Collect all samples and group by exact length
    for i in range(len(dataset)):
        sample = dataset[i]
        seq_len = sample['seq_len']
        
        # Group by exact length - no tolerance
        length_groups[seq_len].append(i)
    
    return length_groups


def create_no_padding_collate_fn():
    """Create a collate function that doesn't pad sequences"""
    
    def collate_fn(batch):
        """Custom collate function without padding - all sequences should have same length"""
        question_ids = [item['question_id'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        image_sizes = [item['image_size'] for item in batch]
        
        # Stack input_ids directly (no padding needed as they have same length)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        
        # Stack image tensors
        image_tensors = torch.stack([item['image_tensor'] for item in batch])
        
        # No attention mask needed since there's no padding
        return {
            'question_ids': question_ids,
            'input_ids': input_ids,
            'image_tensors': image_tensors,
            'image_sizes': image_sizes,
            'prompts': prompts
        }
    
    return collate_fn


class LengthGroupedDataLoader:
    """Custom DataLoader that groups samples by exact length to avoid padding"""
    
    def __init__(self, dataset, batch_size, min_group_size=1, shuffle_groups=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_group_size = min_group_size
        self.shuffle_groups = shuffle_groups
        
        # Group samples by exact length
        self.length_groups = group_by_exact_length(dataset)
        
        # Create batches for each exact length group
        self.batches = []
        total_samples = 0
        
        for length_key, indices in self.length_groups.items():
            # Only create batches if we have enough samples in this length group
            if len(indices) >= self.min_group_size:
                if self.shuffle_groups:
                    import random
                    random.shuffle(indices)
                
                # Split indices into batches
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    self.batches.append(batch_indices)
                    total_samples += len(batch_indices)
            else:
                # For small groups, process individually (fallback to single sample batches)
                for idx in indices:
                    self.batches.append([idx])
                    total_samples += 1
        
        print(f"Created {len(self.batches)} batches from {len(self.length_groups)} exact length groups")
        print(f"Total samples to process: {total_samples}")
        
        # Print length distribution
        for length_key, indices in sorted(self.length_groups.items()):
            if len(indices) >= self.min_group_size:
                num_batches = math.ceil(len(indices) / batch_size)
                print(f"Length {length_key}: {len(indices)} samples, {num_batches} batches")
            else:
                print(f"Length {length_key}: {len(indices)} samples, {len(indices)} single-sample batches")
    
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        collate_fn = create_no_padding_collate_fn()
        
        for batch_indices in self.batches:
            # Get samples for this batch
            batch_samples = [self.dataset[i] for i in batch_indices]
            
            # Apply collate function
            batch = collate_fn(batch_samples)
            yield batch


def eval_model_batch(args):
    """Main function for batch inference without padding"""
    
    # Initialize model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    # Load and split questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Resume support: skip already-processed questions
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    completed_ids = set()
    if args.resume and os.path.isfile(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                    completed_ids.add(result["question_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Resume mode: found {len(completed_ids)} already completed questions")
        original_count = len(questions)
        questions = [q for q in questions if q["question_id"] not in completed_ids]
        print(f"Resume mode: {original_count - len(questions)} skipped, {len(questions)} remaining")
        
        if len(questions) == 0:
            print("All questions already processed. Nothing to do.")
            return
    
    # Create dataset
    dataset = LLaVADataset(
        questions, 
        args.image_folder, 
        image_processor, 
        model.config, 
        tokenizer, 
        args.conv_mode
    )
    
    # Create length-grouped dataloader (no padding needed)
    dataloader = LengthGroupedDataLoader(
        dataset, 
        batch_size=args.batch_size,
        min_group_size=args.min_group_size,
        shuffle_groups=False
    )
    
    # Open file in append mode when resuming, write mode otherwise
    file_mode = "a" if args.resume and len(completed_ids) > 0 else "w"
    ans_file = open(answers_file, file_mode)
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Starting batch inference with batch_size={args.batch_size}")
    print(f"Total batches: {len(dataloader)}")
    print(f"Total samples: {len(dataset)}")
    print("No padding will be applied - sequences grouped by exact length")
    
    # Batch inference loop
    with torch.inference_mode():
        processed_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            question_ids = batch['question_ids']
            input_ids = batch['input_ids'].cuda()
            image_tensors = batch['image_tensors'].half().cuda()
            image_sizes = batch['image_sizes']
            prompts = batch['prompts']
            
            # Generate responses (no attention_mask needed since no padding)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )

            # Decode outputs
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Process and save results
            for i, (question_id, prompt, output) in enumerate(zip(question_ids, prompts, outputs)):
                # Clean the output
                output = output.strip()
                
                # Remove input prompt from output if it's included
                if prompt in output:
                    output = output.replace(prompt, "").strip()
                
                # Generate unique answer ID
                ans_id = shortuuid.uuid()
                
                # Create result dictionary
                result = {
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": output,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {}
                }
                
                # Write to file
                ans_file.write(json.dumps(result) + "\n")
            
            # Flush to ensure data is written
            ans_file.flush()
            
            # Update progress counter
            processed_samples += len(question_ids)
            
            # Optional: Print progress
            if (batch_idx + 1) % args.progress_interval == 0:
                print(f"Processed {processed_samples} samples")
    
    ans_file.close()
    print(f"Batch inference completed. Results saved to {answers_file}")
    print(f"Total processed samples: {processed_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Batch Inference without Padding")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
                       help="Path to the model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path")
    
    # Data arguments
    parser.add_argument("--image-folder", type=str, default="",
                       help="Path to the image folder")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl",
                       help="Path to the question file")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl",
                       help="Path to save answers")
    
    # Conversation arguments
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="Conversation mode")
    
    # Distributed processing arguments
    parser.add_argument("--num-chunks", type=int, default=1,
                       help="Number of chunks for distributed processing")
    parser.add_argument("--chunk-idx", type=int, default=0,
                       help="Index of current chunk")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p for generation")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="Number of beams for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1025,
                       help="Maximum new tokens to generate")
    
    # Batch processing arguments
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--min-group-size", type=int, default=2,
                       help="Minimum number of samples needed to form a batch (others processed individually)")
    
    # Monitoring arguments
    parser.add_argument("--progress-interval", type=int, default=10,
                        help="Print progress every N batches")
    
    # Resume arguments
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing answers file, skipping already processed questions")
    
    args = parser.parse_args()
    
    eval_model_batch(args)