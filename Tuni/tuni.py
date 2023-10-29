from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tensorflow as tf
import math
import EdgeGPT
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
from pydub import AudioSegment
import openai
from EdgeGPT import Chatbot, ConversationStyle
r = sr.Recognizer()
import os
import asyncio
import re
import keyboard

"""
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large",padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
"""

step = 0
chat_history_ids = None
new_user_input_ids = None
user_hist = []
bot_hist = []
bot = Chatbot()
asyncio.run(bot.ask(prompt="My name is ____ _____", conversation_style=ConversationStyle.precise))
def rem(inp):
    try:
        inp = inp.replace("*","")
        for i in range(10):
            inp = inp.replace(f"[^{i}^]","")
        return inp
    except:
        return "fail"
async def main():
    global step
    global chat_history_ids
    global user_hist
    global bot_hist
    global bot
    with sr.Microphone() as source:
        voice = None
        print("-------------")
        print("Say something!",end="\r")
        audio = r.listen(source)
        print("Responding!        ",end="\r")
        try:
            voice = r.recognize_google(audio,language = 'en-IN', show_all = True)['alternative'][0]['transcript']
            run = True
        except:
            run = False
            voice = "No input detected"
        print(f"You: {voice}")
        if "bing hard reset" == voice.lower():
            bot = Chatbot()
            run = False
            print("Sys: Bot Sucsessfully Reset")
        if voice.lower() == "exit":
            print("Terminating Program")
            exit()
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    if run == True:
        """
        gt = voice
        # append the new user input tokens to the chat history
        new_user_input_ids = tokenizer.encode(gt + tokenizer.eos_token, return_tensors='pt')
        user_hist.append(new_user_input_ids)
        bot_input_ids = torch.cat(user_hist[:], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(
            bot_input_ids, 
            do_sample=True, 
            max_length=math.inf, 
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.5
        )
        bot_hist.append(chat_history_ids)
        #print(chat_history_ids)
        if int(chat_history_ids[:, bot_input_ids.shape[-1]:][0][0]) == 0:
            print("no response")
            resp = "No comment"
        # generated a response while limiting the total chat history to 1000 tokens, 

        # activate sampling and deactivate top_k by setting top_k sampling to 0

        # pretty print last ouput tokens from bot
        resp = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("DialoGPT: {}".format(resp))
        """
        try: 
            bing_resp = await bot.ask(prompt=voice, conversation_style=ConversationStyle.precise)
            fin = bing_resp['item']['messages'][-1]['text']
            fin = rem(fin)
            print(f"BingAI: {fin}")
        except Exception as err:
            print("Bing failed" + str(err))
        #print(chat_history_ids[:, bot_input_ids.shape[-1]:][0])
        step += 1
        if len(user_hist) > 5:
            user_hist.remove(user_hist[0])
        myobj = gTTS(text=fin, lang='en', slow=False, tld="com.au")
        myobj.save("welcome.mp3")
        playsound('____\\Tuni\\welcome.mp3')
        os.remove('____\\Tuni\\welcome.mp3')
    print("-------------\r")
while True:
    try:
        if keyboard.is_pressed("space"):
            asyncio.run(main())
    except:
        continue 
