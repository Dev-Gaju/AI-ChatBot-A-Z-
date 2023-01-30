from flask import Flask, render_template, request
import torch
import random
import  json, os,nltk
import yagmail
import jinja2
import fcntl
import pdfkit
from datetime import datetime
from Model import NeuralNet, tokenize, bag_of_words

<<<<<<< HEAD
# nltk.download('punkt')
nltk.data.path.append('model/punkt')
=======
nltk.download('punkt')
>>>>>>> 8fed82fda91fe9392f6b87ca015cc61da36cc3b3
url = "model/intents.json"
with open(url, 'r', encoding='utf-8') as f:
    intents = json.load(f)


data = torch.load("model/model.pth")
input_size = data['input_size']
hidden_size = data['hiddent_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html", )

bot_name = "BoT"
replay = " "


@app.route("/get")
def get_response():
    while True:
        global replay
        # global question
        sentence = request.args.get('msg')
        replay += "\n" + sentence + "."
        if sentence == 'quit':
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    answer = bot_name + ": " + random.choice(intent['responses'])
                    # empty.append(answer)
                    replay += "\n" + answer + "."
                    return answer
                    # return str(chatbotjhuuyzykkbvczioo.get_response(userText))
                    # print("{}:{}".format(bot_name,random.choice(intent['responses'])))

        else:
            # print(replay)
            # print(question)
            link = "https://wa.me/contanct_number"
            text = "there"
            html = "<a href='" + link + "' target='_blank' , onclick='myFunction()'>" + text + "</a>"
            return bot_name + ": " + "May be you should Contact..." + " " + html


@app.route("/send_email")
def send_pdf():
    os.remove("model/message.pdf")
    today_date = datetime.today().strftime("%d %b, %Y")
    message=replay
    context= { 'today_date': today_date, 'message': message}
    template_loader = jinja2.FileSystemLoader('./')
    template_env = jinja2.Environment(loader=template_loader)
    html_template = 'templates/basic_template.html'
    template = template_env.get_template(html_template)
    output_text = template.render(context)
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    output_pdf = 'model/message.pdf'
    pdfkit.from_string(output_text, output_pdf, configuration=config, css='templates/style.css')


    # Email and password of the sender
    sender_email = "gmail"
    sender_password = "password"

    # Email of the receiver
    receiver_email = "gajuahmd@gmail.com"

    pdf_file = "model/message.pdf"
    yag = yagmail.SMTP(sender_email, sender_password)
    yag.send(to=receiver_email, subject='Conversion Message ', contents="Please find the attached pdf",
             attachments=pdf_file)
    return 'Email sent'


if __name__ == '__main__':
    app.run()
