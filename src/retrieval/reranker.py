def rerank_results(results):
    for result in results:
        priority = result['priority']
        score = result['score']
        if result['strategy'] == 'sc-cot':
            if 'reasoning_benefit' in result and result['reasoning_benefit'] == 1:
                priority *= 0.9
        result['cost'] = priority * score + 0.6 * score + 0.4 * priority
    return sorted(results, key=lambda x: x['cost'])