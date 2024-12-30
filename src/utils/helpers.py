# src/utils/helpers.py

import logging

def parse_input(input_str: str) -> dict:
    """
    Parses the input string into a dictionary with appropriate data types and encodings.

    Args:
        input_str (str): Input features in the format 'key1=value1,key2=value2,...'

    Returns:
        dict: Dictionary of processed input features.
    """
    input_dict = {}
    for item in input_str.split(','):
        try:
            key, value = item.split('=')
            key = key.strip().lower()
            value = value.strip().lower()
            
            if key == 'sex':
                if value not in ['male', 'female']:
                    raise ValueError(f"Invalid value for sex: {value}. Expected 'male' or 'female'.")
                input_dict['Sex'] = 1 if value == 'female' else 0
            elif key == 'embarked':
                embark_mapping = {'c': 0, 'q': 1, 's': 2}
                if value not in embark_mapping:
                    raise ValueError(f"Invalid value for embarked: {value}. Expected 'C', 'Q', or 'S'.")
                input_dict['Embarked'] = embark_mapping[value]
            elif key == 'class':
                pclass = int(value)
                if pclass not in [1, 2, 3]:
                    raise ValueError(f"Invalid value for class: {pclass}. Expected 1, 2, or 3.")
                input_dict['Pclass'] = pclass
            elif key in ['age', 'fare']:
                numeric_value = float(value)
                if numeric_value < 0:
                    raise ValueError(f"Invalid negative value for {key}: {numeric_value}.")
                input_dict[key.capitalize()] = numeric_value
            elif key in ['sibsp', 'parch']:
                int_value = int(value)
                if int_value < 0:
                    raise ValueError(f"Invalid negative value for {key}: {int_value}.")
                if key == 'sibsp':
                    input_dict['SibSp'] = int_value  # Explicitly map to 'SibSp'
                else:
                    input_dict[key.capitalize()] = int_value  # 'parch' maps to 'Parch'
            else:
                logging.warning(f"Unrecognized feature '{key}' is being ignored.")
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            raise ve
        except Exception as e:
            logging.error(f"Error parsing input item '{item}': {e}")
            raise e
    
    # Assign default values for missing features if necessary
    required_features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for feature in required_features:
        if feature not in input_dict:
            if feature in ['Age', 'Fare']:
                # Assign median values; replace with actual median as per your dataset
                default_value = 30.0 if feature == 'Age' else 32.2042
                input_dict[feature] = default_value
                logging.info(f"Missing '{feature}' assigned default value: {input_dict[feature]}")
            elif feature == 'Sex':
                input_dict[feature] = 0  # Default to 'male'
                logging.info(f"Missing '{feature}' assigned default value: {input_dict[feature]} (male)")
            elif feature == 'Embarked':
                input_dict[feature] = 2  # Default to 'S'
                logging.info(f"Missing '{feature}' assigned default value: {input_dict[feature]} (S)")
            else:
                input_dict[feature] = 0
                logging.info(f"Missing '{feature}' assigned default value: {input_dict[feature]}")
    
    return input_dict
