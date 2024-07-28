import json
import datetime
from flask import Blueprint,url_for,request,render_template,session,redirect
from flask import Flask , render_template , request
from flask_restful import Api , Resource
from markupsafe import Markup
import markdown
import markdown.extensions.fenced_code
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import markdown.extensions.codehilite

Module = Blueprint('Module',__name__)

def get_ai_response(text , model_id):
    ai_response = ''
    try:
        # model_id = 'D:\\pro_of_program\\WebGPT\\model'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

        txt = text

        encode_ids = tokenizer([txt])
        input_ids, attention_mask = torch.LongTensor(encode_ids['input_ids']), torch.LongTensor(
            encode_ids['attention_mask'])

        outs = model.my_generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_seq_len=512,
            search_type='beam',
        )

        outs_txt = tokenizer.batch_decode(outs.cpu().numpy(), skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        ai_response = outs_txt[0]
    except:
        ai_response = '请重新输入'
    if "`" not in ai_response:
        ai_response = ai_response.replace('\n' , '<br />')
    return ai_response

'''
    GET无参数
'''
@Module.route('/apiGETNotParameter' , methods = ["GET"])
def get_Parameters1():
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': None}
    return json.dumps(return_dict, ensure_ascii=False)

"""
    http://127.0.0.1:5000/api/apiGETParameter?text=%27%E4%BD%A0%E5%A5%BD%27
    GET请求，带参数
"""
@Module.route("/apiGETParameter",methods=["GET"])
def get_Parameters2():
    # 默认返回内容
    return_dict = {
        'return_code': '200',
        'return_info': '处理成功',
        'src_result': None, # 原数据
        'ai_response':None
    }
    # 判断入参是否为空
    if len(request.args) == 0:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    # 获取传入的params参数
    get_data = request.args.to_dict()
    text = get_data.get('text')
    return_dict['src_result'] = text
    model_path = "D:\\pro_of_program\\WebGPT\\model"
    return_dict['ai_response'] = get_ai_response(text , model_path)
    return json.dumps(return_dict, ensure_ascii=False)


"""
    POST请求，带参数
"""
@Module.route("/apiPOSTParameter", methods=["POST"])
def get_Parameters3():
    #默认返回内容
    return_dict = {
        'return_code': '200',
        'return_info': '处理成功',
        'src_result': None, # 原数据
        'ai_response':None # ai返回数据
    }
    # 判断传入的json数据是否为空
    if len(request.get_data()) == 0:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    print(request.form.get('text'))
    text = request.form.get('text')
    # 对参数进行操作
    return_dict['src_result'] = text
    model_path = "D:\\pro_of_program\\WebGPT\\model"
    return_dict['ai_response'] = get_ai_response(text , model_path)
    return json.dumps(return_dict,ensure_ascii=False)