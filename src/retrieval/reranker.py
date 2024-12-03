def rerank_results(results):
    for result in results:
        result['cost'] = (result['priority'] + 1) * result['score']
        if result['strategy'] == 'cot-sc':
            if result['reasoning_benefit']:
                result['cost'] *= 0.9
    return sorted(results, key=lambda x: x['cost'])