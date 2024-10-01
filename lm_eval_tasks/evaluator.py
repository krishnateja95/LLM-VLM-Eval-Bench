import collections
import itertools
import numpy as np
import random

import lm_eval
from lm_eval import models
from lm_eval import tasks
from lm_eval import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

def simple_evaluate(
    model,
    model_name,
    tokenizer,
    load='',
    args='',
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    device=None,
    bootstrap_iters=100000,
):

    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"
    
    if 'opt' in model_name:
        lm = lm_eval.models.get_model('opt').create_from_arg_string('', {"model": model, "tokenizer": tokenizer,
                                                                         "batch_size": batch_size, "device": device, 
                                                                         "model_name": model_name, "args":args})
    elif 'bloom' in model_name:
        lm = lm_eval.models.get_model('bloom').create_from_arg_string('', {"model": model, "tokenizer": tokenizer,
                                                                           "batch_size": batch_size, "device": device,
                                                                           "model_name": model_name, "args": args})
    elif 'llama' in model_name:
        lm = models.get_model('llama').create_from_arg_string('', {"model": model, "tokenizer": tokenizer,
                                                                           "batch_size": batch_size, "device": device,
                                                                           "model_name": model_name, "args": args})

    else:
        print("Model not found. Exit")
        exit()
    
        
    task_dict = lm_eval.tasks.get_task_dict(tasks)

    results = evaluate(lm=lm,
                       task_dict=task_dict,
                       num_fewshot=num_fewshot,
                       bootstrap_iters=bootstrap_iters,
                       )

    results["config"] = {
        "model": model_name,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "bootstrap_iters": bootstrap_iters,
    }

    return results



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'ProgramName', description = 'What the program does', epilog = 'Text at the bottom of help')
    parser.add_argument('--model-name', type=str, default='opt')
    parser.add_argument('--task-name', type=str, default='openbookqa')
    parser.add_argument('--num-fewshot', type=int, default=0)
    args = parser.parse_args()

    cache_dir = '/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'


    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, device_map = "auto", cache_dir=cache_dir) 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, cache_dir=cache_dir)

    results = simple_evaluate(model,
                              args.model_name,
                              tokenizer,
                              tasks=[args.task_name],
                              num_fewshot=args.num_fewshot,
                              batch_size=None,
                              device=None)
    
    print(results)


# @positional_deprecated
# def evaluate(
#     lm,
#     task_dict,
#     provide_description=None,
#     num_fewshot=0,
#     limit=None,
#     bootstrap_iters=100000,
#     description_dict=None,
#     decontamination_ngrams_path=None,
# ):
#     """Instantiate and evaluate a model on a list of tasks.

#     :param lm: obj
#         Language Model
#     :param task_dict: dict[str, Task]
#         Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
#     :param provide_description: bool
#         Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
#     :param num_fewshot: int
#         Number of examples in few-shot context
#     :param limit: int, optional
#         Limit the number of examples per task (only use this for testing)
#     :param bootstrap_iters:
#         Number of iterations for bootstrap statistics
#     :param description_dict: dict[str, str]
#         Dictionary of custom task descriptions of the form: `task_name: description`
#     :return
#         Dictionary of results
#     """
#     # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

#     # TODO: todo: implement proper description-providing system
#     assert not provide_description  # not implemented.
#     if provide_description is not None:
#         # nudge people to not specify it at all
#         print(
#             "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
#         )

#     decontaminate = decontamination_ngrams_path is not None

#     task_dict_items = [
#         (name, task)
#         for name, task in task_dict.items()
#         if (task.has_validation_docs() or task.has_test_docs())
#     ]

#     results = collections.defaultdict(dict)
#     versions = collections.defaultdict(dict)

#     requests = collections.defaultdict(list)
#     requests_origin = collections.defaultdict(list)

#     overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

#     # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
#     # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
#     # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
#     #  - probably using an sqlite db because of all the moving parts we have

#     # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
#     docs = {}

#     docs_for_decontamination = collections.defaultdict(list)

#     # get lists of each type of request
#     for task_name, task in task_dict_items:
#         versions[task_name] = task.VERSION
#         # default to test doc, fall back to val doc if validation unavailable
#         # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
#         if task.has_test_docs():
#             task_doc_func = task.test_docs
#             task_set = "test"  # Required for caching in the decontamination
#         elif task.has_validation_docs():
#             task_set = "val"  # Required for caching in the decontamination
#             task_doc_func = task.validation_docs
#         else:
#             raise RuntimeError("Task has neither test_docs nor validation_docs")

#         # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
#         task_docs = list(task_doc_func())
#         rnd = random.Random()
#         rnd.seed(42)
#         rnd.shuffle(task_docs)

#         description = (
#             description_dict[task_name]
#             if description_dict and task_name in description_dict
#             else ""
#         )

#         for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):

#             if decontaminate and task.should_decontaminate():
#                 docs_for_decontamination[(task_name, task_set)].append(
#                     task.doc_to_decontamination_query(doc)
#                 )

#             docs[(task_name, doc_id)] = doc
#             ctx = task.fewshot_context(
#                 doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
#             )
#             reqs = task.construct_requests(doc, ctx)
#             if not isinstance(reqs, (list, tuple)):
#                 reqs = [reqs]
#             for i, req in enumerate(reqs):
#                 requests[req.request_type].append(req)
#                 # i: index in requests for a single task instance
#                 # doc_id: unique id that we can get back to a doc using `docs`
#                 requests_origin[req.request_type].append((i, task_name, doc, doc_id))

#     # Compare all tasks/sets at once to ensure a single training set scan
#     if decontaminate:
#         from lm_eval.decontamination.decontaminate import get_train_overlap

#         print("Finding train/test overlap, please wait...")
#         overlaps = get_train_overlap(
#             docs_for_decontamination, decontamination_ngrams_path, limit
#         )

#     # all responses for each (task, doc)
#     process_res_queue = collections.defaultdict(list)

#     # execute each type of request
#     for reqtype, reqs in requests.items():
#         # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
#         #       only in index. We could implement some kind of caching, but that would be more of a band-aid
#         #       solution. we could also implement some kind of auto-grouping here;
#         #       they should end up next to each other.

#         print("Running", reqtype, "requests")
#         # code.interact(banner='interact', local=locals())
#         resps = getattr(lm, reqtype)([req.args for req in reqs])
#         resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]

#         for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
#             process_res_queue[(task_name, doc_id)].append((i, resp))

#     vals = collections.defaultdict(list)

#     # unpack results and sort back in order and return control to Task
#     for (task_name, doc_id), requests in process_res_queue.items():
#         requests.sort(key=lambda x: x[0])
#         requests = [x[1] for x in requests]

#         task = task_dict[task_name]
#         doc = docs[(task_name, doc_id)]

#         metrics = task.process_results(doc, requests)
#         for metric, value in metrics.items():
#             vals[(task_name, metric)].append(value)

#             # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
#             if decontaminate and task_name in overlaps:
#                 if doc_id not in overlaps[task_name]:
#                     vals[(task_name, metric + decontaminate_suffix)].append(value)

#     # aggregate results
#     for (task_name, metric), items in vals.items():
#         task = task_dict[task_name]
#         real_metric = metric  # key when looking up the metric with task.aggregation
#         if metric.endswith(decontaminate_suffix):
#             real_metric = metric.replace(
#                 decontaminate_suffix, ""
#             )  # decontaminated still uses the same metric
#         results[task_name][metric] = task.aggregation()[real_metric](items)

#         # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
#         # so we run them less iterations. still looking for a cleaner way to do this

#         stderr = lm_eval.metrics.stderr_for_metric(
#             metric=task.aggregation()[real_metric],
#             bootstrap_iters=min(bootstrap_iters, 1000)
#             if metric in ["bleu", "chrf", "ter"]
#             else bootstrap_iters,
#         )

#         if stderr is not None:
#             results[task_name][metric + "_stderr"] = stderr(items)

#     return {"results": dict(results), "versions": dict(versions)}


# def make_table(result_dict):
#     """Generate table of results."""
#     from pytablewriter import MarkdownTableWriter, LatexTableWriter

#     md_writer = MarkdownTableWriter()
#     latex_writer = LatexTableWriter()
#     md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
#     latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

#     values = []

#     for k, dic in result_dict["results"].items():
#         version = result_dict["versions"][k]
#         for m, v in dic.items():
#             if m.endswith("_stderr"):
#                 continue

#             if m + "_stderr" in dic:
#                 se = dic[m + "_stderr"]
#                 values.append([k, version, m, "%.8f" % v, "±", "%.8f" % se])
#             else:
#                 values.append([k, version, m, "%.8f" % v, "", ""])
#             k = ""
#             version = ""
#     md_writer.value_matrix = values
#     latex_writer.value_matrix = values

#     # todo: make latex table look good
#     # print(latex_writer.dumps())

#     return md_writer.dumps()
