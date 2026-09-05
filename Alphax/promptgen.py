import json
import utils
import config

def operators():
    with open('operators_detailed.json', 'r') as f:
        operators = json.load(f)

    operator_prompt = ''

    for operator in operators:
        operator_prompt += operator['definition']
        operator_prompt += ':\n'

        # summary = operator.get('summary')

        # if (summary):
        #     operator_prompt += summary
        # else:
        #     operator_prompt += operator['description']
        #     operator_prompt += '\n'
        # operator_prompt += '\n'
        operator_prompt += operator['description']
        operator_prompt += '\n\n'

    print(operator_prompt.strip())


def prompt_with_fields():
    data_fields = ''

    try:
        with open('prompts/prompt.txt', 'r') as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        with open('prompt.txt', 'r') as f:
            prompt = f.read().strip()
    
    sum = 0
    for data_set_id in utils.data_set.evergreen:
        with open(f'fields/evergreen/{data_set_id}.json') as f:
            data_set = json.load(f)
        
        sum += len(data_set)

    for data_set_id in utils.data_set.evergreen:
        with open(f'fields/evergreen/{data_set_id}.json') as f:
            data_set = json.load(f)
        
        sample_count = max(1, int((len(data_set) * 100 + sum - 1) / sum))
        sample_count = min(sample_count, len(data_set))  # clamp to dataset length

        for i in range(sample_count):
            try:
                field = data_set[i]
                field_type = field['type']
                field_id = field['id']
                
                if field_type == 'VECTOR':
                    data_fields += f"{field_id} (VECTOR): {field['description']}"
                if field_type == 'MATRIX':
                    data_fields += f"{field_id} (MATRIX): {field['description']}"
                if field_type == 'GROUP':
                    data_fields += f"{field_id} (GROUP): {field['description']}"
                data_fields += '\n'
            except (IndexError, KeyError) as e:
                print(f'Warning: skipping field at index {i} in {data_set_id}: {e}')
                continue

    prompt = prompt.replace('{data_fields_substitute}', data_fields.strip())
    prompt = prompt.replace('{sharpe_threshold}', str(config.sharpe_threshold))
    prompt = prompt.replace('{fitness_threshold}', str(config.fitness_threshold))

    print(prompt)

    return prompt

if __name__ == '__main__':
    operators()
    prompt_with_fields()