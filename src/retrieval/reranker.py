def rerank_results(results):
    """
    Reranks search results based on both similarity score and strategy priority,
    using a confidence-based approach.
    
    Args:
        results: List of dictionaries containing search results with 'priority', 'score',
                and 'strategy' fields
    
    Returns:
        Sorted list of results based on calculated confidence score
    """
    for result in results:
        # Base confidence is average of priority and similarity score
        base_confidence = (result['priority'] + result['score']) / 2
        
        scaled_confidence = base_confidence / 5        
        if result['strategy'] == 'sc-cot':
            scaled_confidence *= 1.15
            
        scaled_confidence = min(scaled_confidence, 1.0)
        result['confidence'] = scaled_confidence
        
    return sorted(results, key=lambda x: x['confidence'], reverse=True)