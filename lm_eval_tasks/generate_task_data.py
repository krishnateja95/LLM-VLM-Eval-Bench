import argparse
import json

from lm_eval import tasks
import json

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--output-file', type=str, default='input.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--num-fewshot', type=int, default=0)
    args = parser.parse_args()

    from lm_eval import evaluator, tasks, models

    # Define the model to evaluate
    model = models.get_model("hf")("gpt2")  # Using GPT-2 as an example

    

    # Define the tasks to evaluate on
    task_names = ["lambada_openai", "hellaswag"]

    # Initialize the tasks
    task_dict = tasks.get_task_dict(task_names)

    # Run the evaluation
    results = evaluator.evaluate(model, task_dict, num_fewshot=0, batch_size=1)

    # Print the results
    print(evaluator.make_table(results))













    task_manager = tasks.TaskManager()

    lm_eval_task = task_manager.get_task(args.task_name)

    lm_eval_task.download()

    test_data = lm_eval_task.dataset["test"]

    output_data = []
    for item in test_data:
        output_item = {
            "question": item["question"],
            "choices": item["choices"]["text"],
            "answer": item["choices"]["text"][item["choices"]["label"].index(item["answer"])]
        }
        output_data.append(output_item)

    output_file = "openbookqa_zero_shot.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"{args.task_name} dataset for zero-shot evaluation has been downloaded and written to {output_file}")

    # seq = 1024
    # total_batch = 1
    # pe = 'fixed'

    # with open(args.output_file, 'w') as f:
    #     pass

    # class DryRunner:
    #     def eval(self, batch):

    #         with open(args.output_file, 'a') as f:

    #             for text in batch['text']:
    #                 item = {
    #                     "best_of": 1, 
    #                     "echo": True, 
    #                     "logprobs": 1, 
    #                     "max_tokens": 0, 
    #                     "model": "x", 
    #                     "n": 1, 
    #                     "prompt": text, 
    #                     "request_type": "language-model-inference", 
    #                     "stop": None, 
    #                     "temperature": 0, 
    #                     "top_p": 1
    #                 }

    #                 f.write(json.dumps(item) + '\n')

    #         out = {
    #             'mask_loss': [1.0] * len(batch),
    #             'each_correct': [True] * len(batch),
    #         }
    #         return out

    # t = DryRunner()
    # adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")
    # results = evaluator.evaluate(adaptor, tasks.get_task_dict([args.task_name
    #                                                         #"lambada_openai",
    #                                                         #"piqa",
    #                                                         #"hellaswag",
    #                                                         #"winogrande",
    #                                                         #"mathqa",
    #                                                         #"pubmedqa",
    #                                                         # "boolq",
    #                                                         # "cb",
    #                                                         # "copa",
    #                                                         # "multirc",
    #                                                         # "record",
    #                                                         # "wic",
    #                                                         # "wsc",
    #                                                         ]), False, args.num_fewshot, None)
    # print('Finished')