import json


def explanation_guided_filtering():
    with open('tasks.json', 'r') as f:
        data = json.load(f)

    difficulties = {}

    for d in data:
        filename = d['metadata']['filename']
        image_id = filename.split('_')[3].lstrip('0')
        if 'response' in d:
            annotations = d['response'].get('annotations')
            if annotations:
                relevance_rating = annotations.get('Relevance Rating')
                if relevance_rating:
                    if image_id in difficulties:
                        difficulties[image_id].append(float(relevance_rating))
                    else:
                        difficulties[image_id] = [float(relevance_rating)]

    for image_id, diff_list in difficulties.items():
        if len(diff_list) > 0:
            diff_sum = sum(diff_list)
            # 映射到0-1之间
            difficulty = diff_sum / (len(diff_list) * 5)
            difficulties[image_id] = difficulty

    # 将图片的困难值按照降序排序，并打印结果(注意越接近0表示越困难)
    sorted_difficulties = sorted(difficulties.items(), key=lambda x: x[1], reverse=True)
    for image_id, difficulty in sorted_difficulties:
        print(f'image_id: {image_id}, difficulty: {difficulty:.2f}')


if __name__ == '__main__':
    explanation_guided_filtering()


