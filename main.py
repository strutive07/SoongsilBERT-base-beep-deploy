import sentencepiece

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import logging

from flask import Flask, request, render_template, jsonify
import torch
from torch.nn import functional as F
import traceback

import os
from queue import Queue, Empty
from threading import Thread
import time

import re
import emoji
from soynlp.normalizer import repeat_normalize

logger = logging.getLogger(__file__)

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
urlpattern = re.compile(
    r'https?://(www.)?[-a-zA-Z0-9@:%.+~#=]{1,256}.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&//=]*)')

def clean_text(x):
    x = pattern.sub(' ', x)
    x = urlpattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def remove_suffix(row):
    if row["title"][:len(row["category"])] == row["category"]:
        row["title"] = row["title"][len(row["category"]):]
    return row

category_map = {
    "0": "no push",
    "1": "push",
}

category_map_logits = {
    "0": "no push",
    "1": "push",
}

os.system('ls')
app = Flask(__name__)

tokenizer = RobertaTokenizer.from_pretrained('jason9693/SoongsilBERT-notice-base')
model = RobertaForSequenceClassification.from_pretrained('jason9693/SoongsilBERT-notice-base')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
app.logger.info("Model load finished")


requests_queue = Queue()    # request queue.
BATCH_SIZE = 100              # max request size.
CHECK_INTERVAL = 0.1

##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []
        text_list = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request = requests_queue.get(timeout=CHECK_INTERVAL)
                request_batch.append(request)
            except Empty:
                break

        if len(request_batch) == 0:
           continue
        # outputs = mk_predict(text_list)
        valid_requests = []
        valid_texts = []
        for idx, request in enumerate(request_batch):
            types = request["input"][0]
            txt = request["input"][1]
            category = request["input"][2]
            
            txt = remove_suffix({'title': txt, 'category': category})['title']
            txt = clean_text(txt)
            
            valid_texts.append(txt)
            valid_requests.append(request)
        request_batch = []
            # except Exception as e:
            #     request["output"] = e
            #     return
        
        outputs = mk_predict(valid_texts)[0]
        for idx, request in enumerate(valid_requests):
            try:
                dpstring = []
                output_item = outputs[idx]
                if request["input"][0] == "logits":
                    return_item = {
                            category_map_logits[str(k)]:v for k, v 
                            in enumerate(output_item.softmax(-1).tolist())}
                elif request["input"][0] == "dplogits":
                    return_item = {
                            category_map[str(k)]:v for k, v 
                            in enumerate(output_item.softmax(-1).tolist())}
                else:
                    return_item = str(torch.argmax(output_item, -1).item())
                    dpstring = category_map[return_item]
                if request["input"][0] == "dpclass":
                    request["output"] = ({0: category_map[return_item]}, 200)
                elif request["input"][0] == "dplogits":
                    request["output"] = ({0: '<br>'.join(
                        [f"{k}: {v:.4f}" for k, v in return_item.items()])}, 200)
                else:
                    request["output"] = {
                        "result": return_item,
                        "dpstring": dpstring
                    }
            except Exception as e:
                request["output"] = e
    return


handler = Thread(target=handle_requests_by_batch).start()


##
# GPT-2 generator.
def mk_predict(text_array: list):
    try:
        inputs = tokenizer(text_array, return_tensors="pt")
        outputs = model(**inputs)[0]

        return outputs, 200

    except Exception as e:
        traceback.print_exc()
        return {'error': e}, 500


##
# Get post request page.
@app.route('/predict/<types>', methods=['POST'])
def generate(types):
    if types not in ['logits', 'class', 'dplogits', 'dpclass']:
        return {'Error': 'Invalid types'}, 404

    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return {'Error': 'Too Many Requests'}, 429

    try:
        args = []

        text = request.form['text']
        category = request.form.get('category', '')

        args.append(types)
        args.append(text)
        args.append(category)

    except Exception as e:
        logger.exception(exc_info=e)
        return {'message': 'Invalid request'}, 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']

##
# Get post request page.
@app.route('/predict_json/<types>', methods=['POST'])
def generate(types):
    if types not in ['logits', 'class', 'dplogits', 'dpclass']:
        return {'Error': 'Invalid types'}, 404

    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return {'Error': 'Too Many Requests'}, 429

    try:
        args = []
        req_json = request.get_json(force=True)

        text = req_json['text']
        category = req_json('category', '')

        args.append(types)
        args.append(text)
        args.append(category)

    except Exception as e:
        logger.exception(exc_info=e)
        return {'message': 'Invalid request'}, 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


##
# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200


##
# Sever health checking page.
@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


##
# Main page.
@app.route('/')
def main():
    return render_template('index.html'), 200


if __name__ == '__main__':
    from waitress import serve
    app.logger.info("server start")
    serve(app, port=80, host='0.0.0.0')
