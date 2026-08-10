import logging

def MKCL_loss_logging(loss):
    logging.info('='*80)
    logging.info('MKCL Loss Configuration Initialized')
    logging.info('='*80)
    logging.info(f'Loss Component Weights:')
    logging.info(f'  └─ Multi-Knowledge Contrastive Loss (lambda_m): {loss.lambda_m}')
    logging.info(f'  └─ Subcaption Local Region Alignment Loss (lambda_s): {loss.lambda_s}')
    logging.info(f'  └─ use_disease_specific_weight = {loss.use_disease_specific_weight}')
    logging.info(f'  └─ Loss Type: {loss.loss_type}')
    logging.info(f'')
    logging.info(f'Knowledge Configuration:')
    logging.info(f'  └─ Number of subcaptions: {loss.num_subcaption}')
    logging.info(f'  └─ Total caption types: {3 + loss.num_subcaption} (original + ontology + visual_concept + subcaptions)')
    logging.info(f'  └─ OHCL Temperature: {loss.temp}')
    logging.info(f'  └─ OHCL soft raitio beta: {loss.beta}')