from flask import Flask , render_template , request
from flask_restful import Api , Resource
from markupsafe import Markup
import markdown
import markdown.extensions.fenced_code
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from controller.Flask_api import Module
import torch
import markdown.extensions.codehilite

app = Flask(__name__)
app.register_blueprint(Module , url_prefix = '/api')

messages = []
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST' , 'GET'])
def get_bot_response():
    user_input = request.form['user_input']
    # print(user_input)
    messages.append({'role': 'user', 'content': user_input})
    model_id = 'D:\\pro_of_program\\WebGPT\\model'

    ai_response = ''
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

        txt = user_input

        encode_ids = tokenizer([txt])
        input_ids, attention_mask = torch.LongTensor(encode_ids['input_ids']), torch.LongTensor(
            encode_ids['attention_mask'])

        outs = model.my_generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_seq_len=512,
            search_type='beam',
        )

        outs_txt = tokenizer.batch_decode(outs.cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ai_response = outs_txt[0]
    except:
        ai_response = '请重新输入'
    if "`" not in ai_response:
        ai_response = ai_response.replace('\n' , '<br />')
    # '<br />'
    # ai_response.strip('\n')
    # print(ai_response)
    messages.append({'role': 'assistant', 'content': ai_response})
    print(messages)
    return  Markup(markdown.markdown(ai_response, extensions=['fenced_code', 'codehilite']))

@app.route('/reset')
def reset():
    global messages
    messages = []
    return "Conversation history has been reset."

@app.route('/api/doc')
def api_doc():
    return render_template('api.html')

if __name__ == '__main__':
    app.run()