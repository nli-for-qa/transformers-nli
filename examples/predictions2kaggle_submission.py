import  argparse
import json
import pandas as pd


def create_short_answer(entry):
    answer = []
    if entry['answer_type'] == 0:
        return ""

    elif entry['answer_type'] == 1:
        return 'YES'

    elif entry['answer_type'] == 2:
        return 'NO'

    elif entry["short_answers_score"] < 10.15567662:
        return ""

    else:
        for short_answer in entry["short_answers"]:
            if short_answer["start_token"] > -1:
                answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))

        return " ".join(answer)


def create_long_answer(entry):
    answer = []

    if entry['answer_type'] == 0:
        return ''

    elif entry["long_answer_score"] < 8.2944288:
        return ""

    elif entry["long_answer"]["start_token"] > -1:
        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))
        return " ".join(answer)

def read_pred(pred_file):
    with open(pred_file, 'r') as f:
        predictions = json.loads(f.read())
    return predictions['predictions']

def convert(preds, output_file):
    nq_pred_dict = {}
    for example in preds:
        nq_pred_dict[example['example_id']] = example
        if not example.get('answer_type', None):
            example['answer_type'] = 3

    predictions_json = {"predictions": list(nq_pred_dict.values())}
    with open(output_file, 'w') as f:
        json.dump(predictions_json, f, indent=4)
    test_answers_df = pd.read_json(output_file)
    for var_name in ['long_answer_score', 'short_answers_score', 'answer_type']:
        test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])
    test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)
    test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)
    test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))

    long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))
    short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))

    return  long_answers, short_answers

def write(sub_file, long_answers, short_answers):
    sample_submission = pd.read_csv(sub_file)

    long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(
        lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)
    short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(
        lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)

    sample_submission.loc[
        sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings
    sample_submission.loc[
        sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings

    sample_submission.to_csv(sub_file, index=False)


def convert_predictions2submissions(pred_file, sub_file):
    preds = read_pred(pred_file)
    long_answers, short_answers = convert(preds, sub_file + '_predictions_json')
    write(sub_file, long_answers, short_answers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True, help='predictions json from run squad')
    parser.add_argument('--sub_file', required=True, help='submissions file to kaggle')
    args = parser.parse_args()
    convert_predictions2submissions(args.pred_file, args.sub_file)
