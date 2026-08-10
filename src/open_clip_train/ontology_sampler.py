import random
import numpy as np
import logging
from collections import defaultdict

class GraphBasedConnectivitySampler:
    """
    Minimal connectivity sampler focused on core objectives:
    - 50% Knowledge Transfer (Head → Tail)
    - 50% Peer Learning (Tail ↔ Tail)
    - Minimal hyperparameters
    - Maximum efficiency
    """
    
    def __init__(self, dataset, batch_size, similarity_matrix_path, 
                 max_epochs=15, tail_threshold=0.3, temperature=0.07):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tail_threshold = tail_threshold
        self.temperature = temperature
        
        # Load similarity matrix
        self.similarity_matrix = np.load(similarity_matrix_path)
        np.fill_diagonal(self.similarity_matrix, 1.0)
        
        self.connectivity_weight_schedule = self._build_cosine_schedule()
        self._build_indices_and_labels()
        self._identify_head_tail_classes()
        self._compute_natural_distribution()
        
        self.current_epoch = 0
        logging.info(f"Minimal connectivity sampler: temperature={temperature}")
        logging.info(f"Head classes: {len(self.head_classes)}, Tail classes: {len(self.tail_classes)}")
        logging.info(f"Natural head ratio: {self.natural_head_ratio:.3f}")
    
    def _build_cosine_schedule(self):
        """Build cosine schedule for connectivity weight (0 → 1)"""
        schedule = []
        for epoch in range(self.max_epochs):
            # Cosine annealing from 0 to 1
            progress = epoch / (self.max_epochs - 1)
            weight = 0.5 * (1 - np.cos(np.pi * progress))
            schedule.append(weight)
        return schedule

    def get_current_connectivity_weight(self):
        """Get current connectivity weight based on epoch"""
        if self.current_epoch >= len(self.connectivity_weight_schedule):
            return 1.0
        return self.connectivity_weight_schedule[self.current_epoch]
    
    def _build_indices_and_labels(self):
        """Build basic mappings"""
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.label_cache = {}
        
        for idx in range(len(self.dataset)):
            if self.dataset.ontology_target[idx] != -1:
                self.labeled_indices.append(idx)
                self.label_cache[idx] = int(self.dataset.ontology_target[idx])
            else:
                self.unlabeled_indices.append(idx)
        
        self.labeled_set = set(self.labeled_indices)
        total_samples = len(self.labeled_indices) + len(self.unlabeled_indices)
        self.labeled_ratio = len(self.labeled_indices) / total_samples
        self.labeled_per_batch = int(self.batch_size * self.labeled_ratio)
        self.unlabeled_per_batch = self.batch_size - self.labeled_per_batch
        
        # Group samples by label
        self.samples_by_label = defaultdict(list)
        for idx in self.labeled_indices:
            label = self.label_cache[idx]
            self.samples_by_label[label].append(idx)
        
        self.all_labels = list(self.samples_by_label.keys())
    
    def _identify_head_tail_classes(self):
        """Identify head and tail classes"""
        class_counts = [(label, len(self.samples_by_label[label])) for label in self.all_labels]
        class_counts.sort(key=lambda x: x[1])
        
        total_samples = len(self.labeled_indices)
        target_tail_samples = int(total_samples * self.tail_threshold)
        
        self.tail_classes = set()
        current_tail_samples = 0
        
        for label, count in class_counts:
            if current_tail_samples + count <= target_tail_samples:
                self.tail_classes.add(label)
                current_tail_samples += count
            else:
                break
        
        self.head_classes = set(self.all_labels) - self.tail_classes
        
        self.head_samples = [idx for idx in self.labeled_indices 
                           if self.label_cache[idx] in self.head_classes]
        self.tail_samples = [idx for idx in self.labeled_indices 
                           if self.label_cache[idx] in self.tail_classes]
    
    def _compute_natural_distribution(self):
        """Compute natural distribution ratios"""
        total_labeled = len(self.labeled_indices)
        self.natural_head_ratio = len(self.head_samples) / total_labeled
        self.natural_tail_ratio = len(self.tail_samples) / total_labeled
    
    def get_similarity(self, label1, label2):
        """Get similarity between two labels"""
        try:
            return self.similarity_matrix[label1, label2]
        except IndexError:
            return 0.0
    
    def calculate_connectivity_score(self, labeled_samples):
        """
        Core connectivity score: 50% knowledge transfer + 50% peer learning
        """
        if len(labeled_samples) < 2:
            return 0.0
        
        sample_labels = [self.label_cache[idx] for idx in labeled_samples]
        unique_labels = list(set(sample_labels))
        
        if len(unique_labels) < 2:
            return 0.0
        
        head_labels = [label for label in unique_labels if label in self.head_classes]
        tail_labels = [label for label in unique_labels if label in self.tail_classes]
        
        total_score = 0.0
        
        # 1. Knowledge Transfer Score (Head → Tail): 50%
        if head_labels and tail_labels:
            transfer_scores = []
            for tail_label in tail_labels:
                max_transfer = max(self.get_similarity(tail_label, head_label) 
                                 for head_label in head_labels)
                transfer_scores.append(max_transfer)
            
            knowledge_transfer_score = np.mean(transfer_scores)
            total_score += 0.5 * knowledge_transfer_score
        
        # 2. Peer Learning Score (Tail ↔ Tail): 50%
        if len(tail_labels) > 1:
            peer_scores = []
            for i, tail1 in enumerate(tail_labels):
                max_peer = 0.0
                for j, tail2 in enumerate(tail_labels):
                    if i != j:
                        sim = self.get_similarity(tail1, tail2)
                        max_peer = max(max_peer, sim)
                peer_scores.append(max_peer)
            
            peer_learning_score = np.mean(peer_scores)
            total_score += 0.5 * peer_learning_score
        
        return total_score
    
    def create_connectivity_batch(self, available_labeled, available_unlabeled):
        """Create batch optimized for connectivity"""
        batch = []
        
        # Add unlabeled samples
        unlabeled_needed = min(self.unlabeled_per_batch, len(available_unlabeled))
        if unlabeled_needed > 0:
            selected_unlabeled = random.sample(available_unlabeled, unlabeled_needed)
            batch.extend(selected_unlabeled)
            for idx in selected_unlabeled:
                available_unlabeled.remove(idx)
        
        # Connectivity-optimized labeled sample selection
        labeled_needed = min(self.labeled_per_batch, len(available_labeled))
        if labeled_needed > 0:
            selected_labeled = self._connectivity_labeled_selection(available_labeled, labeled_needed)
            batch.extend(selected_labeled)
            for idx in selected_labeled:
                available_labeled.remove(idx)
        
        return batch
    
    def _connectivity_labeled_selection(self, available_labeled, needed):
        """
        Select labeled samples with mixed strategy based on epoch
        """
        if needed >= len(available_labeled):
            return available_labeled.copy()

        connectivity_weight = self.get_current_connectivity_weight()
        
        # Split needed samples between strategies
        connectivity_needed = int(needed * connectivity_weight)
        random_needed = needed - connectivity_needed

        selected_samples = []

        # Random selection first (early epoch dominant)
        if random_needed > 0:
            random_selected = random.sample(available_labeled, 
                                          min(random_needed, len(available_labeled)))
            selected_samples.extend(random_selected)
            # Remove from available pool
            available_for_connectivity = [idx for idx in available_labeled 
                                        if idx not in random_selected]
        else:
            available_for_connectivity = available_labeled

        # Connectivity-based selection (late epoch dominant)
        if connectivity_needed > 0 and available_for_connectivity:
            connectivity_selected = self._pure_connectivity_selection(
                available_for_connectivity, connectivity_needed, selected_samples
            )
            selected_samples.extend(connectivity_selected)

        return selected_samples
    
    def _pure_connectivity_selection(self, available_labeled, needed, already_selected):
        """
        Pure connectivity-based selection
        """
        if needed >= len(available_labeled):
            return available_labeled.copy()
    
        # Separate head and tail samples
        available_head = [idx for idx in available_labeled if self.label_cache[idx] in self.head_classes]
        available_tail = [idx for idx in available_labeled if self.label_cache[idx] in self.tail_classes]
    
        # Get already selected labels
        already_selected_labels = [self.label_cache[idx] for idx in already_selected]
    
        # Determine split based on natural distribution
        target_head_count = int(needed * self.natural_head_ratio)
        target_tail_count = needed - target_head_count
    
        # Adjust for availability
        actual_head_count = min(target_head_count, len(available_head))
        actual_tail_count = min(target_tail_count, len(available_tail))
    
        # Redistribute remaining slots
        remaining_slots = needed - actual_head_count - actual_tail_count
        if remaining_slots > 0:
            if len(available_head) > actual_head_count:
                additional_head = min(remaining_slots, len(available_head) - actual_head_count)
                actual_head_count += additional_head
                remaining_slots -= additional_head

            if remaining_slots > 0 and len(available_tail) > actual_tail_count:
                actual_tail_count += remaining_slots
    
        selected_samples = []
    
        # Select head samples (can be random since we focus on tail connectivity)
        if actual_head_count > 0 and available_head:
            selected_head = random.sample(available_head, actual_head_count)
            selected_samples.extend(selected_head)
    
        # Connectivity-optimized tail selection
        if actual_tail_count > 0 and available_tail:
            current_labels = already_selected_labels + [self.label_cache[idx] for idx in selected_samples]
            selected_tail = self._optimized_tail_selection(
                available_tail, actual_tail_count, current_labels
            )
            selected_samples.extend(selected_tail)
    
        return selected_samples
    
    def _optimized_tail_selection(self, available_tail, needed, already_selected_labels):
        """
        Optimized tail selection using connectivity scores
        """
        if needed >= len(available_tail):
            return available_tail.copy()
        
        # Group tail samples by label
        tail_by_label = defaultdict(list)
        for idx in available_tail:
            label = self.label_cache[idx]
            tail_by_label[label].append(idx)
        
        available_tail_labels = list(tail_by_label.keys())
        selected_tail = []
        current_labels = set(already_selected_labels)
        
        # Iterative selection based on connectivity
        for _ in range(needed):
            if not available_tail_labels:
                break
            
            # Calculate connectivity scores for all available tail labels
            scores = []
            valid_labels = []
            
            for tail_label in available_tail_labels:
                if not tail_by_label[tail_label]:
                    continue
                
                connectivity_score = self._calculate_tail_connectivity(
                    tail_label, current_labels, available_tail_labels
                )
                
                scores.append(connectivity_score)
                valid_labels.append(tail_label)
            
            if not valid_labels:
                break
            
            # Probabilistic selection based on connectivity scores
            selected_label = self._probabilistic_selection(valid_labels, scores)
            
            if selected_label and tail_by_label[selected_label]:
                sample = random.choice(tail_by_label[selected_label])
                selected_tail.append(sample)
                current_labels.add(selected_label)
                tail_by_label[selected_label].remove(sample)
                
                if not tail_by_label[selected_label]:
                    available_tail_labels.remove(selected_label)
        
        return selected_tail
    
    def _calculate_tail_connectivity(self, tail_label, current_labels, available_tail_labels):
        """
        Calculate connectivity score for a tail label
        50% knowledge transfer (to heads) + 50% peer learning (to other tails)
        """
        score = 0.0
        
        # Knowledge transfer component (50%)
        head_labels_in_current = [label for label in current_labels if label in self.head_classes]
        if head_labels_in_current:
            max_head_sim = max(self.get_similarity(tail_label, head_label) 
                             for head_label in head_labels_in_current)
            score += 0.5 * max_head_sim
        
        # Peer learning component (50%)
        tail_labels_available = [label for label in available_tail_labels 
                               if label != tail_label]
        if tail_labels_available:
            max_tail_sim = max(self.get_similarity(tail_label, other_tail) 
                             for other_tail in tail_labels_available)
            score += 0.5 * max_tail_sim
        
        return score
    
    def _probabilistic_selection(self, candidates, scores):
        """
        Probabilistic selection with temperature
        """
        if not candidates or not scores:
            return random.choice(candidates) if candidates else None
        
        scores = np.array(scores)
        
        if self.temperature <= 0 or all(s == 0 for s in scores):
            # Deterministic selection
            best_idx = np.argmax(scores)
            return candidates[best_idx]
        
        # Temperature scaling and softmax
        scaled_scores = scores / self.temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)  # Numerical stability
        
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)
        
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[selected_idx]
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def __iter__(self):
        """Generate samples for one epoch with minimal connectivity optimization"""
        available_labeled = self.labeled_indices.copy()
        available_unlabeled = self.unlabeled_indices.copy()
        
        all_batches = []
        connectivity_scores = []
        
        while available_labeled or available_unlabeled:
            if len(available_labeled) + len(available_unlabeled) < self.batch_size:
                remaining = available_labeled + available_unlabeled
                if remaining:
                    random.shuffle(remaining)
                    all_batches.append(remaining)
                break
            
            batch = self.create_connectivity_batch(available_labeled, available_unlabeled)
            if batch:
                all_batches.append(batch)
                
                # Calculate connectivity score
                labeled_in_batch = [idx for idx in batch if idx in self.labeled_set]
                if len(labeled_in_batch) >= 2:
                    score = self.calculate_connectivity_score(labeled_in_batch)
                    connectivity_scores.append(score)
        
        # Log statistics
        if connectivity_scores:
            mean_score = np.mean(connectivity_scores)
            logging.info(f"Epoch {self.current_epoch}: connectivity_score={mean_score:.4f}")
        
        random.shuffle(all_batches)
        all_indices = []
        for batch in all_batches:
            random.shuffle(batch)
            all_indices.extend(batch)
        
        self.current_epoch += 1
        return iter(all_indices)
    
    def __len__(self):
        return len(self.labeled_indices) + len(self.unlabeled_indices)