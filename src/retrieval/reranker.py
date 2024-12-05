def rerank_results_old(results):
    for result in results:
        priority = result['priority']
        score = result['score']
        if result['strategy'] == 'sc-cot':
            if 'reasoning_benefit' in result and result['reasoning_benefit'] == 1:
                priority *= 0.9
        result['cost'] = priority * score + 0.6 * score + 0.4 * priority
    return sorted(results, key=lambda x: x['cost'])

def rerank_results(results):
    for result in results:
        base_confidence = (result['priority'] + result['score']) / 2        
        scaled_confidence = base_confidence / 5        
        if result['strategy'] == 'sc-cot':
            scaled_confidence *= 1.15
            
        scaled_confidence = min(scaled_confidence, 1.0)
        result['cost'] = 1 - scaled_confidence
        result['confidence'] = scaled_confidence
    return sorted(results, key=lambda x: x['confidence'], reverse=True)