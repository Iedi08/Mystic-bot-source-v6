from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import math
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import asyncio
import string
import datetime
import random
from collections import Counter
import discord
import json
import cloudscraper
import os 
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from discord.app_commands import Choice
from discord import app_commands
import http.client
import os
import sklearn

# Change to the directory where your script or necessary files are located
os.chdir('/path/to/your/directory')

# Your existing bot code
import discord
from discord.ext import commands

# Initialize the bot
bot = commands.Bot(command_prefix='!')

# Define bot commands and events here

# Start the bot
bot.run('YOUR_BOT_TOKEN')





scraper = cloudscraper.create_scraper()
dir = 'C:/Users/admin/Desktop/mystic v6'






current_path = os.path.dirname(os.path.abspath(__file__))
print("The directory where this script is located:", current_path)



class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print(f"We have logged in as {self.user}.")


client = aclient()
tree = app_commands.CommandTree(client)
customerrole = 'bot customer' 

with open(f"{dir}/tokens.json", "r") as f:
    auths = json.load(f)



with open(f"{dir}/Predcounts.json", "r") as f:
    lilbro = json.load(f)



with open(f"{dir}/betamnts.json", "r") as f:
    bigbro = json.load(f)

with open(f"{dir}/ursigma.json", "r") as f:
    sigma = json.load(f)

with open(f"{dir}/urskibidi.json", "r") as f:
    skibiditoilet = json.load(f)

with open(f"{dir}/urohio.json", "r") as f:
    ohio = json.load(f)

                
@tree.command(name='link', description='|✅| Attach Your Bloxflip Account To Our API For Predictions. (Mystic V6)')
async def link(interaction: discord.Interaction, authtoken: str):
    user_id = interaction.user.id
    user = interaction.user
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket` in the purchase channel.', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return

    original = discord.Embed(title=f'> ```💥``` {user} We are currently linking you to `Mystic` API. ', description='Wait 2 or 3 seconds so we can `complete` the process...', color=discord.Color.orange())
    original.set_image(url='https://media.giphy.com/media/KB1ezhDIi1iC8j8G94/giphy.gif?cid=ecf05e4702i8po7ihndxd86hopswqdsqx0icskjh2teu4bax&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    original.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=original, ephemeral=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
        'X-Auth-Token': authtoken
    }

    conn = http.client.HTTPSConnection("api.bloxflip.com")
    conn.request("GET", "/user", "", headers)

    response = conn.getresponse()
    data = json.loads(response.read().decode('utf-8'))

    if not data['success']:
        embed = discord.Embed(title="> ```❌```We were unable to link your account to `Mystic`.", description="This issue occured because of the following:", color=discord.Color.red())
        embed.add_field(name=f'> `💥` The provided `authentication token` is invalid.', value='Your provided `auth` token was invalid and try again with a correct one.', inline=False)
        embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        embed.set_footer(text='💥 Mystic V6')
        await interaction.edit_original_response(embed=embed)
    else:
        if str(interaction.user.id) in auths:
            auths[str(interaction.user.id)]["auth_token"] = authtoken
        else:

            auths[str(interaction.user.id)] = {"auth_token": authtoken}
        with open(f"{dir}/tokens.json", "w") as f:
            json.dump(auths, f)
                
        id = data["user"]["robloxId"]
        print(data)
        name = data["user"]["robloxUsername"]


        
        url = f"https://thumbnails.roblox.com/v1/users/avatar-headshot?userIds={id}&size=352x352&format=Png&isCircular=false"
        response = scraper.get(url)
        user_pfp = response.json()["data"][0]["imageUrl"]
        member = interaction.user
        embed = discord.Embed(title=f'> `💥` {user} The Process Was a success, thank\'s for choosing `Mystic V6`', description='Thanks to your patience, we sucessfully linked your bloxflip account to `Mystic` Paid. Enjoy!', color=discord.Color.orange())
        embed.add_field(name='> `🗣️` Bloxflip & Roblox Username:', value=f'`{name}`')
        embed.add_field(name='> `🛂` Bloxflip & Roblox User ID:', value=f'`{id}`')
        embed.add_field(name='> `😆` Discord User ID:', value=f'`{user_id}`', inline=False)
        embed.add_field(name='> `😆` Discord Username:', value=f'`{user}`')
        embed.add_field(name='> `✅` Honorable Mentions:', value=f'`Check out 3vil and other good communities!`', inline=False)
        embed.set_image(url=f'{user_pfp}')
        if str(interaction.user.id) in auths:
            embed.add_field(name='> ```💥``` By the way!', value='not a while ago you already linked to Mystic predictions, but!!! we changed the old authentication token with this new one, so have fun! :D')
        embed.set_thumbnail(url=f'https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGR0NW1qeTYyZXRjOGk3YTM4NjU1d3ZrZzlxaXpvYmdyejFxOWY1YyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
        embed.set_image(url=f'{user_pfp}')
        embed.set_footer(text=f'💥 Mystic V6')
        await interaction.edit_original_response(embed=embed)


import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

def LootFriendlyFire(safe_amount, interaction):
    user_tokens = auths.get(str(interaction.user.id))
    num_history_games = 50
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }

            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("GET", f"/games/mines/history?size={num_history_games}&page=0", "", headers)

            response = conn.getresponse()
            
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                
                features = []
                labels = []
                for game in game_data:
                    mines = game['mineLocations']
                    cell_counts = [1 if i in mines else 0 for i in range(25)]
                    features.append(cell_counts)
                    labels.append(cell_counts) 
                
                features = np.array(features)
                labels = np.array(labels)

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
                model = xgb.XGBClassifier(random_state=0)
                model.fit(X_train, y_train)
                
                next_game_features = np.zeros((1, 25))
                next_game_features_scaled = scaler.transform(next_game_features)
                predictions = model.predict_proba(next_game_features_scaled)[0]
                
                predicted_safe_spots = np.argsort(predictions)[-safe_amount:]
                prediction = '\n'.join(''.join(['🔶' if (i * 5 + j) in predicted_safe_spots else '❌' for j in range(5)]) for i in range(5))
                
                return prediction
            else:
                return "ERROR WITH PREDICTION REQUEST"

    

def OldPast(safe_amount, interaction):
    user_tokens = auths.get(str(interaction.user.id))
    num_history_games = 50
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("GET", f"/games/mines/history?size={num_history_games}&page=0", "", headers)

            response = conn.getresponse()
            
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                
                features = []
                labels = []
                for game in game_data:
                    mines = game['mineLocations']
                    cell_counts = [1 if i in mines else 0 for i in range(25)]
                    features.append(cell_counts)
                    labels.append(cell_counts) 
                
                features = np.array(features)
                labels = np.array(labels)

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

                models = []
                for i in range(25):
                    model = xgb.XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss')
                    model.fit(X_train, y_train[:, i])
                    models.append(model)
                
                next_game_features = np.zeros((1, 25))
                next_game_features_scaled = scaler.transform(next_game_features)
                predictions = np.array([model.predict_proba(next_game_features_scaled)[0, 1] for model in models])
                
                predicted_safe_spots = np.argsort(predictions)[:safe_amount]

                prediction_grid = [['❌' for _ in range(5)] for _ in range(5)]
                for cell in predicted_safe_spots:
                    row, col = divmod(cell, 5)
                    prediction_grid[row][col] = '🔶'
                
                prediction = '\n'.join(''.join(row) for row in prediction_grid)
                
                return prediction



import requests
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def ArtificialAnalyze(safe_amount, interaction):
    user_tokens = auths.get(str(interaction.user.id))
    num_history_games = 50

    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("GET", f"/games/mines/history?size={num_history_games}&page=0", "", headers)

            response = conn.getresponse()
            
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']

                mine_probabilities = [0] * 25
                for game in game_data:
                    mines = game['mineLocations']
                    for mine in mines:
                        mine_probabilities[mine] += 1
                
                total_games = len(game_data)
                mine_probabilities = [count / total_games for count in mine_probabilities]
                
                safe_probabilities = [1 - prob for prob in mine_probabilities]

                sorted_indices = np.argsort(safe_probabilities)[::-1]
                predicted_safe_spots = sorted_indices[:safe_amount]

                prediction = '\n'.join(''.join(['🔶' if (i * 5 + j) in predicted_safe_spots else '❌' for j in range(5)]) for i in range(5))

                return prediction



import numpy as np
import requests
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def predict_safe_spots(game_data, safe_amount):
    cell_counts = [0] * 25
    for game in game_data:
        mines = game['mineLocations']
        for mine in mines:
            cell_counts[mine] += 1
    sorted_cells = sorted(range(len(cell_counts)), key=lambda k: cell_counts[k])
    predicted_safe_spots = sorted_cells[:safe_amount]

    return predicted_safe_spots



def CellLookup(safe_amount, interaction):
    user_tokens = auths.get(str(interaction.user.id))
    num_history_games = 5
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("GET", f"/games/mines/history?size={num_history_games}&page=0", "", headers)

            response = conn.getresponse()
            
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                
                predicted_safe_spots = predict_safe_spots(game_data, safe_amount)
                
                prediction = '\n'.join(''.join(['🔶' if (i * 5 + j) in predicted_safe_spots else '❌' for j in range(5)]) for i in range(5))
                
                return prediction


            



def HistoryFeast(safe_amount, interaction):
    user_tokens = auths.get(str(interaction.user.id))
    num_history_games = 5
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }
            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("GET", f"/games/mines/history?size={num_history_games}&page=0", "", headers)

            response = conn.getresponse()
            
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                mine_counts = np.zeros(25)
                
                for game in game_data:
                    mines = game['mineLocations']
                    for mine in mines:
                        mine_counts[mine] += 1
                

                mine_probabilities = mine_counts / num_history_games
            
                safe_probabilities = 1 - mine_probabilities
                safest_cells = np.argsort(safe_probabilities)[-safe_amount:]
                prediction_grid = [['❌' for _ in range(5)] for _ in range(5)]
                for cell in safest_cells:
                    row, col = divmod(cell, 5)
                    prediction_grid[row][col] = '🔶'

                prediction = '\n'.join(''.join(row) for row in prediction_grid)
                
                return prediction













with open(f"{dir}/realjson.json", "r") as f:
    realjson = json.load(f)
    print(realjson)

@app_commands.choices(method=[
    Choice(name="1. LootFriendlyFire", value="LootFriendlyFire"),
    Choice(name="2. OldPast", value="OldPast"),
    Choice(name="3. Artificial Analyze", value="ArtificialAnalyze"),
    Choice(name="4. CellLookup", value="CellLookup"),
    Choice(name="5. HistoryFeast", value="HistoryFeast")

])



    

@tree.command(name='setmines_method', description='|📚| Set Your Minesweeper Prediction Method For Bloxflip Using Our API. (Mystic V6)')
async def mines(interaction: discord.Interaction, method: str):
    user = interaction.user
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title=f'> ```🔥``` The Command was a success.', description=f'We Have Sucessfully set your `prediction` method to `{method},` **{user}**', color=discord.Color.orange())
    embed.add_field(name=f'> ```🎮``` {user} Alongside.', value=f'Worry not. mystic predictor `saves` every single prediction `method` in our database, offering an experience where you won\'t need to set the same method again and again.')
    embed.add_field(name=f'> ```😆``` Enjoy!', value=f'Go, bro! make hella `ROBUX` with Mystic\'s amazing `Prediction` methods!')    
    embed.set_thumbnail(url='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzF0bnlxM3BlYmc5Z3N3dHFtcjl4a2l5aDd0bHd1bDZtbGZrdmxydyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
    embed.set_image(url='https://media.giphy.com/media/fBGjcpJBKwFTkUktR3/giphy.gif?cid=790b761150765wo2aj6zf48qwx39jjl4elejs4hny2hz9f3i&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text=f'💥 Mystic V6')
    if str(interaction.user.id) in realjson:
        realjson[str(interaction.user.id)]["minesmethod"] = method
        embed.add_field(name='> ```👁️``` About your old prediction method:', value=f'We changed your old prediction method to: `{method}`, the one you requested just now.')
    else:
        realjson[str(interaction.user.id)] = {"minesmethod": method}
    with open(f"{dir}/realjson.json", "w") as f:
        json.dump(realjson, f)
    await interaction.response.send_message(embed=embed)


@tree.command(name='mines', description='|🦒| Predict Your Bloxflip Minesweeper Game Using Our API. (Mystic V6)')
async def mines(interaction: discord.Interaction, safe_amount: int):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    
    methods = realjson.get(str(interaction.user.id))
    if methods:
        method = methods.get("minesmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        Ot.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.`', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        emberror.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=emberror)
        return
    embed = discord.Embed(title='> ```💥``` Generating Minesweeper Prediction Using `Mystic`...', description=f'> `🔥` Please wait patiently for your `prediction..`', color=discord.Color.orange())
    embed.set_image(url='https://media.giphy.com/media/Uy2fAyrf5UCWTtSBT0/giphy.gif?cid=ecf05e47ocuugf6ajz3nh4ni6cem6d3nc8nyx50e2pxb9p5h&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)
    user_tokens = auths.get(str(interaction.user.id))
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            conn = http.client.HTTPSConnection("api.bloxflip.com")

            payload = ""

            headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                'X-Auth-Token': auth_token,
                }
            

            conn.request("GET", "/games/mines", payload, headers)

            res = conn.getresponse()
            data = json.loads(res.read().decode('utf-8'))

            if not data['hasGame']:
                embed = discord.Embed(title='> ```❌``` Sir. you must start a game to predict!', description="Rerun this command afterwards.", color=discord.Color.red())
                embed.set_footer(text='💥 Mystic V6')

                embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
                await interaction.edit_original_response(embed=embed)
            else:
                bombs = data['game']['minesAmount']
                roundid = data['game']['uuid']
                betamnt = data['game']['betAmount']
                hash = data['game']['_id']['$oid']

    
    if method == 'LootFriendlyFire':
        prediction = LootFriendlyFire(safe_amount, interaction)
    elif method == 'OldPast':
        prediction = OldPast(safe_amount, interaction)

    elif method == 'ArtificialAnalyze':
        prediction = ArtificialAnalyze(safe_amount, interaction)
    elif method == 'CellLookup':
        prediction = CellLookup(safe_amount, interaction)
    elif method == 'HistoryFeast':
        prediction = HistoryFeast(safe_amount, interaction)


    if str(interaction.user.id) in lilbro:
        lilbro[str(interaction.user.id)]["userpredamount"] += 1
    else:
        lilbro[str(interaction.user.id)] = {"userpredamount": 1}

    with open(f"{dir}/Predcounts.json", "w") as f:
        json.dump(lilbro, f)

    if str(interaction.user.id) in bigbro:
        bigbro[str(interaction.user.id)]["userbetamount"] += betamnt
    else:
        bigbro[str(interaction.user.id)] = {"userbetamount": betamnt}

    with open(f"{dir}/betamnts.json", "w") as f:
        json.dump(bigbro, f)



    embed1 = discord.Embed(title='> ```💥``` Mystic Predictions `V6.`', description=f'{user}. Let\'s Be a bit straightforward.' + '\n' + ' here is your `prediction:`', color=discord.Color.orange())
    embed1.add_field(name='Prediction:', value=f'{prediction}', inline=False)
    embed1.add_field(name='> `🆔` Round ID:', value=f'{roundid}', inline=False)
    embed1.add_field(name='> `🥢` Hash:', value=f'{hash}', inline=False)
    embed1.add_field(name='> `💣` Bombs:', value=f'{bombs}.0', inline=False)
    embed1.add_field(name='> `✔️` Safe:', value=f'{safe_amount}.0', inline=False)
    embed1.add_field(name='> `💳` Bet:', value=f'{betamnt}', inline=False)
    embed1.add_field(name=f'> `🎮` Prediction method:', value=f'{method}', inline=False)
    embed1.add_field(name=f'> `💢` Not Profiting?', value=f'{user}, we attempt to predict your future game, the predictor' + '\n' + 'is not 100% `accurate.`' + '\n' + 'make sure you don\'t risk too much and also use `unrig.`', inline=False)
    embed1.set_footer(text='💥 Mystic V6')
    await interaction.edit_original_response(embed=embed1)

    









    return safe_amount, interaction








import requests
import numpy as np

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def Zephyr(interaction, num_rows=8, num_columns=3):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            conn = http.client.HTTPSConnection("api.bloxflip.com")

            payload = ""

            headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                'X-Auth-Token': auth_token,
                }

            conn.request("GET", "/games/towers/history?size=20&page=0", payload, headers)

            response = conn.getresponse()
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                tower_levels = [game['towerLevels'] for game in game_data]
                flattened_data = []
                for level in tower_levels:
                   flattened_data.extend(level)

                #Pad shorter sublists with zeros
                max_len = max(len(sublist) for sublist in flattened_data)
                flattened_data = [sublist + [1] * (max_len - len(sublist)) for sublist in flattened_data]

                X = np.array(flattened_data)
                y = np.argmin(X, axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model Accuracy: {accuracy}")
                prediction = []
                for i in range(num_rows):
                    r = np.array([flattened_data[i][j] for j in range(num_columns)]).reshape(1,-1)
                    row_prediction = clf.predict(r)
                    prediction.append(['🔶' if col == row_prediction[0] else '❌' for col in range(num_columns)])
                
                prediction_text = '\n'.join([''.join(row) for row in prediction])
                return prediction_text
            else:
                return None
    return None




def WhateverBlue(interaction, num_rows=8, num_columns=3):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            conn = http.client.HTTPSConnection("api.bloxflip.com")

            payload = ""

            headers = {
                'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                'X-Auth-Token': auth_token,
                }

            conn.request("GET", "/games/towers/history?size=20&page=0", payload, headers)

            response = conn.getresponse()
            if response.status == 200:
                game_data = json.loads(response.read().decode('utf-8'))['data']
                tower_levels = [game['towerLevels'] for game in game_data]
                flattened_data = []
                for level in tower_levels:
                   flattened_data.extend(level)

                #Pad shorter sublists with zeros
                max_len = max(len(sublist) for sublist in flattened_data)
                flattened_data = [sublist + [1] * (max_len - len(sublist)) for sublist in flattened_data]

                X = np.array(flattened_data)
                y = np.argmin(X, axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model Accuracy: {accuracy}")
                prediction = []
                for i in range(num_rows):
                    r = np.array([flattened_data[i][j] for j in range(num_columns)]).reshape(1,-1)
                    row_prediction = clf.predict(r)
                    prediction.append(['🔶' if col == row_prediction[0] else '❌' for col in range(num_columns)])
                
                prediction_text = '\n'.join([''.join(row) for row in prediction])
                return prediction_text
            else:
                return None
    return None

def Detroit(interaction, num_rows=8, num_columns=3):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            try:
                conn = http.client.HTTPSConnection("api.bloxflip.com")

                payload = ""

                headers = {
                    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    'X-Auth-Token': auth_token,
                    }

                conn.request("GET", "/games/towers/history?size=20&page=0", payload, headers)

                response = conn.getresponse()

                if response.status == 200:
                    game_data = json.loads(response.read().decode('utf-8'))['data']
                    tower_levels = [game['towerLevels'] for game in game_data]
                    prediction = []

                    for i in range(num_rows):
                        row_prediction = ['❌' for _ in range(num_columns)]
                        bomb_counts = [sum(level[i][j] if j < len(level[i]) else 1 for level in tower_levels) for j in range(num_columns)]
                        safest_column = min(range(len(bomb_counts)), key=bomb_counts.__getitem__)
                        row_prediction[safest_column] = '🔶'

                        prediction.append(row_prediction)
                    
                    prediction_text = '\n'.join([''.join(row) for row in prediction])
                    return prediction_text
                else:
                    return None
            except Exception as e:
                print(f"Error occurred: {e}")
                return None
    return None



def BombTorture(interaction, num_rows=8, num_columns=3):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            try:
                conn = http.client.HTTPSConnection("api.bloxflip.com")

                payload = ""

                headers = {
                    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    'X-Auth-Token': auth_token,
                    }

                conn.request("GET", "/games/towers/history?size=1&page=0", payload, headers)

                response = conn.getresponse()
                
                if response.status == 200:
                    game_data = json.loads(response.read().decode('utf-8'))['data']
                    tower_levels = [game['towerLevels'] for game in game_data]
                    prediction = []

                    for i in range(num_rows):
                        row_prediction = ['❌' for _ in range(num_columns)]
                        bomb_counts = [sum(level[i][j] if j < len(level[i]) else 1 for level in tower_levels) for j in range(num_columns)]
                        safest_column = max(range(len(bomb_counts)), key=bomb_counts.__getitem__)
                        row_prediction[safest_column] = '🔶'

                        prediction.append(row_prediction)
                    
                    prediction_text = '\n'.join([''.join(row) for row in prediction])
                    return prediction_text
                else:
                    return None
            except Exception as e:
                print(f"Error occurred: {e}")
                return None
    return None


def SafeRetreat(interaction, num_rows=8, num_columns=3):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            try:
                conn = http.client.HTTPSConnection("api.bloxflip.com")

                payload = ""

                headers = {
                    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                    'X-Auth-Token': auth_token,
                    }

                conn.request("GET", "/games/towers/history?size=5&page=0", payload, headers)

                response = conn.getresponse()
                
                if response.status == 200:
                    game_data = json.loads(response.read().decode('utf-8'))['data']
                    tower_levels = [game['towerLevels'] for game in game_data]
                    prediction = []

                    for i in range(num_rows):
                        row_prediction = ['❌' for _ in range(num_columns)]
                        bomb_counts = [sum(level[i][j] if j < len(level[i]) else 1 for level in tower_levels) for j in range(num_columns)]
                        median_index = len(bomb_counts) // 2
                        median_column = sorted(range(len(bomb_counts)), key=lambda x: bomb_counts[x])[median_index]
                        
                        row_prediction[median_column] = '🔶'
                        prediction.append(row_prediction)
                    
                    prediction_text = '\n'.join([''.join(row) for row in prediction])
                    return prediction_text
                else:
                    return None
            except Exception as e:
                print(f"Error occurred: {e}")
                return None
    return None







@tree.command(name='towers', description='|🗼| Predict Your Bloxflip Towers Game Using Our API. (Mystic V6)')
async def towers(interaction: discord.Interaction):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    

    methods = realjson.get(str(interaction.user.id))
    if methods:
        method = methods.get("towersmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        Ot.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.`', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        emberror.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=emberror)
        return
    embed = discord.Embed(title='> ```💥``` Generating Tower Prediction Using `Mystic`...', description=f'> `🔥` Please wait patiently for your `prediction..`', color=discord.Color.orange())
    embed.set_image(url='https://media.giphy.com/media/Uy2fAyrf5UCWTtSBT0/giphy.gif?cid=ecf05e47ocuugf6ajz3nh4ni6cem6d3nc8nyx50e2pxb9p5h&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)
    time.sleep(2)
    user_tokens = auths.get(str(interaction.user.id))
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }

            conn = http.client.HTTPSConnection("api.bloxflip.com")
            payload = ""

            conn.request("GET", "/games/towers", payload, headers)

            res = conn.getresponse()
            data = json.loads(res.read().decode('utf-8'))

            if not data['hasGame']:
                embed = discord.Embed(title='> ```❌``` Sir. you must start a game to predict!', description="Rerun this command afterwards.", color=discord.Color.red())
                embed.set_footer(text='💥 Mystic V6')

                embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
                await interaction.edit_original_response(embed=embed)
            else:
                difficulty = data['game']['difficulty']
                roundid = data['game']['uuid']
                betamnt = data['game']['betAmount']
                hash = data['game']['_id']['$oid']

    if method == 'Zephyr':
        prediction = Zephyr(interaction)
    elif method == 'WhateverBlue':
        prediction = WhateverBlue(interaction)
    elif method == 'Detroit':
        prediction = Detroit(interaction)
    elif method == 'BombTorture':
        prediction = BombTorture(interaction)
    elif method == 'SafeRetreat':
        prediction = SafeRetreat(interaction)



    if str(interaction.user.id) in lilbro:
        lilbro[str(interaction.user.id)]["userpredamount"] += 1
    else:
        lilbro[str(interaction.user.id)] = {"userpredamount": 1}

    with open(f"{dir}/Predcounts.json", "w") as f:
        json.dump(lilbro, f)


    if str(interaction.user.id) in bigbro:
        bigbro[str(interaction.user.id)]["userbetamount"] += betamnt
    else:
        bigbro[str(interaction.user.id)] = {"userbetamount": betamnt}

    with open(f"{dir}/betamnts.json", "w") as f:
        json.dump(bigbro, f)




    embed1 = discord.Embed(title='> ```💥```  `V6.`', description=f'{user}. Let\'s Be a bit straightforward.' + '\n' + ' here is your `prediction:`', color=discord.Color.orange())
    embed1.add_field(name='Prediction:', value=f'{prediction}', inline=False)
    embed1.add_field(name='> `🆔` Round ID:', value=f'{roundid}', inline=False)
    embed1.add_field(name='> `🥢` Hash:', value=f'{hash}', inline=False)
    embed1.add_field(name='> `💢` Difficulty:', value=f'{difficulty}.0', inline=False)
    embed1.add_field(name='> `💳` Bet:', value=f'{betamnt}', inline=False)
    embed1.add_field(name=f'> `🎮` Prediction method:', value=f'{method}', inline=False)
    embed1.add_field(name=f'> `💢` Not Profiting?', value=f'{user}, we attempt to predict your future game, the predictor' + '\n' + 'is not 100% `accurate.`' + '\n' + 'make sure you don\'t risk too much and also use `unrig.`', inline=False)
    embed1.set_footer(text='💥 ')
    await interaction.edit_original_response(embed=embed1)




    





    return interaction








@app_commands.choices(method=[
    Choice(name="1. Zephyr", value="Zephyr"),
    Choice(name="2. WhateverBlue", value="WhateverBlue"),
    Choice(name="3. Detroit", value="Detroit"),
    Choice(name="4. BombTorture", value="BombTorture"),
    Choice(name="5. Safe Retreat", value="SafeRetreat")
])


@tree.command(name='settowers_method', description='|🔖| Set Your Towers Prediction Method For Bloxflip Using Our API. (Mystic V6)')
async def mines(interaction: discord.Interaction, method: str):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title=f'> ```🔥``` The Command was a success.', description=f'We Have Sucessfully set your `prediction` method to `{method},` **{user}**', color=discord.Color.orange())
    embed.add_field(name=f'> ```🎮``` {user} Alongside.', value=f'Worry not. mystic predictor `saves` every single prediction `method` in our database, offering an experience where you won\'t need to set the same method again and again.')
    embed.add_field(name=f'> ```😆``` Enjoy!', value=f'Go, bro! make hella `ROBUX` with Mystic\'s amazing `Prediction` methods!')    
    embed.set_thumbnail(url='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzF0bnlxM3BlYmc5Z3N3dHFtcjl4a2l5aDd0bHd1bDZtbGZrdmxydyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
    embed.set_image(url='https://media.giphy.com/media/fBGjcpJBKwFTkUktR3/giphy.gif?cid=790b761150765wo2aj6zf48qwx39jjl4elejs4hny2hz9f3i&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text=f'💥 Mystic V6')
    if str(interaction.user.id) in realjson:
        realjson[str(interaction.user.id)]["towersmethod"] = method
        embed.add_field(name='> ```👁️``` About your old prediction method:', value=f'We changed your old prediction method to: `{method}`, the one you requested just now.')
    else:
        realjson[str(interaction.user.id)] = {"towersmethod": method}
    with open(f"{dir}/realjson.json", "w") as f:
        json.dump(realjson, f)
    await interaction.response.send_message(embed=embed)






def cp(number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
    }
    
    conn = http.client.HTTPSConnection("api.bloxflip.com")
    conn.request("GET", "/games/crash", "", headers)

    response = conn.getresponse()
    data = json.loads(response.read().decode('utf-8'))

    cp = data["history"][number]["crashPoint"]
    return cp



def Voltic():
    data_points = [cp(i) for i in range(1, 11)]
    weights = np.linspace(1, 2, len(data_points)) 
    weighted_mean = np.average(data_points, weights=weights)
    return weighted_mean


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def Average():
    data_points = np.array([cp(i) for i in range(1, 11)])
    X_train = []
    y_train = []
    dick_size = 5

    for i in range(len(data_points) - dick_size):
        X_train.append(data_points[i:i + dick_size])
        y_train.append(data_points[i + dick_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(dick_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, verbose=0)
    last_sequence = data_points[-dick_size:]
    last_sequence = np.reshape(last_sequence, (1, dick_size, 1))
    prediction = model.predict(last_sequence)

    return prediction[0][0]

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def CPTween():
    lastcrashpoint = cp(1)
    lastcrashpoint2 = cp(2)
    lastcrashpoint3 = cp(3)
    lastcrashpoint4 = cp(4)

    X = np.array([[1, lastcrashpoint], [2, lastcrashpoint2], [3, lastcrashpoint3], [4, lastcrashpoint4]])
    y = np.array([lastcrashpoint, lastcrashpoint2, lastcrashpoint3, lastcrashpoint4])
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    test_data = np.array([[2, lastcrashpoint2], [3, lastcrashpoint3], [4, lastcrashpoint4]])
    predictions = model.predict(test_data)
    prediction = predictions[0]
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error Tested on my penis: {mse}")

    return prediction

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def CPFind():
    lastcrashpoint = cp(1)
    lastcrashpoint2 = cp(2)
    lastcrashpoint3 = cp(3)
    lastcrashpoint4 = cp(4)
    X = np.array([[1, lastcrashpoint], [2, lastcrashpoint2], [3, lastcrashpoint3], [4, lastcrashpoint4]])
    y = np.array([lastcrashpoint, lastcrashpoint2, lastcrashpoint3, lastcrashpoint4])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    model.fit(X_train, y_train)

    test_data = np.array([[2, lastcrashpoint2]])
    prediction = model.predict(test_data)[0]
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    print(f"Mean Squared Error on my penis tiktok rizz party: {mse}")

    return prediction




import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def BFTreason():
    lastcrashpoint = cp(1)
    lastcrashpoint2 = cp(2)
    lastcrashpoint3 = cp(3)
    lastcrashpoint4 = cp(4)
    
    X_train = np.array([[1, lastcrashpoint], [2, lastcrashpoint2], [3, lastcrashpoint3], [4, lastcrashpoint4]])
    y_train = np.array([lastcrashpoint, lastcrashpoint2, lastcrashpoint3, lastcrashpoint4])
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    test_data = np.array([[1, lastcrashpoint], [2, lastcrashpoint2], [3, lastcrashpoint3], [3, lastcrashpoint3], [4, lastcrashpoint4]])
    predictions = model.predict(test_data)
    y_pred_train = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred_train)
    print(f"Mean Squared Error on my penis Youtube shorts instagram rizz party skibidi toilet sigma: {mse}")

    return predictions[0]






@app_commands.choices(method=[
    Choice(name="1. Voltic", value="Voltic"),
    Choice(name="2. Average", value="Average"),
    Choice(name="3. CPTween", value="CPTween"),
    Choice(name="4. CPFind", value="CPFind"),
    Choice(name="5. BFTreason", value="BFTreason"),
])
@tree.command(name='setcrash_method', description='|🚀| Set Your Crash Prediction Method For Bloxflip Using Our API. (Mystic V6)')
async def mines(interaction: discord.Interaction, method: str):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title=f'> ```🔥``` The Command was a success.', description=f'We Have Sucessfully set your `prediction` method to `{method},` **{user}**', color=discord.Color.orange())
    embed.add_field(name=f'> ```🎮``` {user} Alongside.', value=f'Worry not. mystic predictor `saves` every single prediction `method` in our database, offering an experience where you won\'t need to set the same method again and again.')
    embed.add_field(name=f'> ```😆``` Enjoy!', value=f'Go, bro! make hella `ROBUX` with Mystic\'s amazing `Prediction` methods!')    
    embed.set_thumbnail(url='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzF0bnlxM3BlYmc5Z3N3dHFtcjl4a2l5aDd0bHd1bDZtbGZrdmxydyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
    embed.set_image(url='https://media.giphy.com/media/fBGjcpJBKwFTkUktR3/giphy.gif?cid=790b761150765wo2aj6zf48qwx39jjl4elejs4hny2hz9f3i&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text=f'💥 Mystic V6')
    if str(interaction.user.id) in realjson:
        realjson[str(interaction.user.id)]["crashmethod"] = method
        embed.add_field(name='> ```👁️``` About your old prediction method:', value=f'We changed your old prediction method to: `{method}`, the one you requested just now.')
    else:
        realjson[str(interaction.user.id)] = {"crashmethod": method}
    with open(f"{dir}/realjson.json", "w") as f:
        json.dump(realjson, f)
    await interaction.response.send_message(embed=embed)




@tree.command(name='crash', description='|🪨| Predict Your Bloxflip Crash Game Using Our API. (Mystic V6)')
async def towers(interaction: discord.Interaction):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    methods = realjson.get(str(interaction.user.id))
    if methods:
        method = methods.get("crashmethod")
    else:
        Ot = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        Ot.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.`', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        emberror.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=emberror)
        return
    embed = discord.Embed(title='> ```💥``` Generating Crash Prediction Using `Mystic`...', description=f'> `🔥` Please wait patiently for your `prediction..`', color=discord.Color.orange())
    embed.set_image(url='https://media.giphy.com/media/Uy2fAyrf5UCWTtSBT0/giphy.gif?cid=ecf05e47ocuugf6ajz3nh4ni6cem6d3nc8nyx50e2pxb9p5h&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)
    time.sleep(2)
    
    if method == 'Voltic':
        prediction = Voltic()
    elif method == 'Average':
        prediction = Average()

    elif method == 'CPTween':
        prediction = CPTween()
    elif method == 'CPFind':
        prediction = CPFind()
    elif method == 'BFTreason':
        prediction = BFTreason()




    if str(interaction.user.id) in lilbro:
        lilbro[str(interaction.user.id)]["userpredamount"] += 1
    else:
        lilbro[str(interaction.user.id)] = {"userpredamount": 1}

    with open(f"{dir}/Predcounts.json", "w") as f:
        json.dump(lilbro, f)



    embed1 = discord.Embed(title='> ```💥``` Mystic Predictions `V6.`', description=f'{user}. Let\'s Be a bit straightforward.' + '\n' + ' here is your `prediction:`', color=discord.Color.orange())
    embed1.add_field(name='Prediction:', value=f'`{prediction}`', inline=False)
    embed1.add_field(name=f'> `🎮` Prediction method:', value=f'{method}', inline=False)
    embed1.add_field(name=f'> `💢` Not Profiting?', value=f'{user}, we attempt to predict your future game, the predictor' + '\n' + 'is not 100% `accurate.`' + '\n' + 'make sure you don\'t risk too much and also use `unrig.`', inline=False)
    embed1.set_footer(text='💥 Mystic V6')
    await interaction.edit_original_response(embed=embed1)




    





    return interaction



@app_commands.choices(method=[
    Choice(name="1. SafeSticky", value="SafeSticky"),
    Choice(name="2. ColourPortfilo", value="ColourPortfilo"),
    Choice(name="3. Safe Spectrum", value="SafeSpectrum"),
    Choice(name="4. ColourNebula", value="ColourNebula"),
    Choice(name="5. LeastColour", value="LeastColour"),
])
@tree.command(name='setslide_method', description='|🛝| Set Your Slide Prediction Method For Bloxflip Using Our API. (Mystic V6)')
async def mines(interaction: discord.Interaction, method: str):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title=f'> ```🔥``` The Command was a success.', description=f'We Have Sucessfully set your `prediction` method to `{method},` **{user}**', color=discord.Color.orange())
    embed.add_field(name=f'> ```🎮``` {user} Alongside.', value=f'Worry not. mystic predictor `saves` every single prediction `method` in our database, offering an experience where you won\'t need to set the same method again and again.')
    embed.add_field(name=f'> ```😆``` Enjoy!', value=f'Go, bro! make hella `ROBUX` with Mystic\'s amazing `Prediction` methods!')    
    embed.set_thumbnail(url='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzF0bnlxM3BlYmc5Z3N3dHFtcjl4a2l5aDd0bHd1bDZtbGZrdmxydyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
    embed.set_image(url='https://media.giphy.com/media/fBGjcpJBKwFTkUktR3/giphy.gif?cid=790b761150765wo2aj6zf48qwx39jjl4elejs4hny2hz9f3i&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text=f'💥 Mystic V6')
    if str(interaction.user.id) in realjson:
        realjson[str(interaction.user.id)]["slidemethod"] = method
        embed.add_field(name='> ```👁️``` About your old prediction method:', value=f'We changed your old prediction method to: `{method}`, the one you requested just now.')
    else:
        realjson[str(interaction.user.id)] = {"slidemethod": method}
    with open(f"{dir}/realjson.json", "w") as f:
        json.dump(realjson, f)
    await interaction.response.send_message(embed=embed)


def winc(interaction, number):
    conn = http.client.HTTPSConnection("api.bloxflip.com")

    payload = ""

    user_tokens = auths.get(str(interaction.user.id))
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
    else:
        return "Prediction error"
    
    if auth_token is None:
        return "Auth failed"

    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/125.0.0.0 Safari/537.36",
        'X-Auth-Token': auth_token,
        }

    conn.request("GET", "/games/roulette", payload, headers)

    response = conn.getresponse()
    data = json.loads(response.read().decode('utf-8'))

    color = data['history'][number]['winningColor']
    return color


def SafeSticky(interaction):
    past = [winc(interaction, i) for i in range(10)]
    color_mapping = {'yellow': 0, 'red': 1, 'purple': 2}
    reverse_color_mapping = {v: k for k, v in color_mapping.items()}
    transition_matrix = np.zeros((3, 3))
    for i in range(1, len(past)):
        current_color = color_mapping[past[i-1]]
        next_color = color_mapping[past[i]]
        transition_matrix[current_color][next_color] += 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    last_color = color_mapping[past[-1]]
    next_color_probabilities = transition_matrix[last_color]
    next_color_encoded = np.argmax(next_color_probabilities)
    next_color = reverse_color_mapping[next_color_encoded]
    
    return next_color


def ColourPortfilo(interaction):
    past = [winc(interaction, i) for i in range(20)] 
    color_counts = {'yellow': 0, 'red': 0, 'purple': 0}
    
    for color in past:
        color_counts[color] += 1
    
    most_frequent_color = max(color_counts, key=color_counts.get)
    
    return most_frequent_color


def SafeSpectrum(interaction):
    past = [winc(interaction, i) for i in range(20)]
    color_sequence = ['red', 'purple'] 
    last_color = past[-1]
    next_color_index = (color_sequence.index(last_color) + 1) % len(color_sequence)
    next_color = color_sequence[next_color_index]
    
    return next_color


def ColourNebula(interaction):
    past = [winc(interaction, i) for i in range(4)]
    color_sequence = ['red', 'purple'] 
    last_color = past[-1]
    second_last_color = past[-2]

    recent_colors = {last_color, second_last_color}
    for color in color_sequence:
        if color not in recent_colors:
            next_color = color
            break
    
    return next_color

from collections import deque
from sklearn.linear_model import LogisticRegression

def LeastColour(interaction):
    past_length = 20
    past = [winc(interaction, i) for i in range(past_length)]
    color_mapping = {'yellow': 0, 'red': 1, 'purple': 2}
    reverse_color_mapping = {v: k for k, v in color_mapping.items()}
    encoded_past = [color_mapping[color] for color in past]
    order = 3
    transition_counts = np.zeros((3**order, 3))
    for i in range(order, past_length):
        previous_states = encoded_past[i-order:i]
        current_state = encoded_past[i]
        index = sum([state * (3**idx) for idx, state in enumerate(reversed(previous_states))])
        transition_counts[index][current_state] += 1

    transition_probabilities = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    transition_probabilities = np.nan_to_num(transition_probabilities)
    last_sequence = encoded_past[-order:]
    last_index = sum([state * (3**idx) for idx, state in enumerate(reversed(last_sequence))])
    next_color_probabilities = transition_probabilities[last_index]
    X = []
    y = []
    for i in range(order, past_length):
        X.append(encoded_past[i-order:i])
        y.append(encoded_past[i])

    model = LogisticRegression()
    model.fit(X, y)
    
    next_state_prediction = model.predict([last_sequence])[0]
    next_color_encoded = next_state_prediction if model.predict_proba([last_sequence]).max() > 0.6 else np.argmax(next_color_probabilities)
    next_color = reverse_color_mapping[next_color_encoded]
    
    return next_color







@tree.command(name='slide', description='|🎲| Predict Your Bloxflip Slide Game Using Our API. (Mystic V6)')
async def towers(interaction: discord.Interaction):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    methods = realjson.get(str(interaction.user.id))
    if methods:
        method = methods.get("slidemethod")
    else:
        Ot = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        Ot.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=Ot)
        return
    if method:
        method = method
    else:
        emberror = discord.Embed(title='> ```❌``` Make Sure To Set Your Prediction Method First!.`', description='Please set your prediction method to procceed with your task.', color=discord.Color.red())
        emberror.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=emberror)
        return
    embed = discord.Embed(title='> ```💥``` Generating Slide Prediction Using `Mystic`...', description=f'> `🔥` Please wait patiently for your `prediction..`', color=discord.Color.orange())
    embed.set_image(url='https://media.giphy.com/media/Uy2fAyrf5UCWTtSBT0/giphy.gif?cid=ecf05e47ocuugf6ajz3nh4ni6cem6d3nc8nyx50e2pxb9p5h&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)
    time.sleep(2)
    
    if method == 'SafeSticky':
        prediction = SafeSticky(interaction)
    elif method == 'ColourPortfilo':
        prediction = ColourPortfilo(interaction)

    elif method == 'SafeSpectrum':
        prediction = SafeSpectrum(interaction)
    elif method == 'ColourNebula':
        prediction = ColourNebula(interaction)
    elif method == 'LeastColour':
        prediction = LeastColour(interaction)



    if str(interaction.user.id) in lilbro:
        lilbro[str(interaction.user.id)]["userpredamount"] += 1
    else:
        lilbro[str(interaction.user.id)] = {"userpredamount": 1}

    with open(f"{dir}/Predcounts.json", "w") as f:
        json.dump(lilbro, f)

    embed1 = discord.Embed(title='> ```💥``` Mystic Predictions `V6.`', description=f'{user}. Let\'s Be a bit straightforward.' + '\n' + ' here is your `prediction:`', color=discord.Color.orange())
    embed1.add_field(name='Prediction:', value=f'`{prediction}`', inline=False)
    embed1.add_field(name=f'> `🎮` Prediction method:', value=f'{method}', inline=False)
    embed1.add_field(name=f'> `💢` Not Profiting?', value=f'{user}, we attempt to predict your future game, the predictor' + '\n' + 'is not 100% `accurate.`' + '\n' + 'make sure you don\'t risk too much and also use `unrig.`', inline=False)
    embed1.set_footer(text='💥 Mystic V6')
    await interaction.edit_original_response(embed=embed1)


@app_commands.choices(time=[
    Choice(name="Lifetime", value=9999),
    Choice(name="Monthly", value=30),
    Choice(name="Weekly", value=7),
])



@tree.command(name="generatekey", description="|🔐| Generate a key for a user using Mystic API. (Mystic V6)")
async def genkey(interaction: discord.Interaction, time: int, quantity: int):
    user = interaction.user
    if "ticket mods" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you don\'t have permission to generate keys of `Mystic`', description=f'If you attempt to generate a key while you don\'t have `permission` won\'t work 🤷‍♂️', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    member = interaction.user
    if quantity == 1:
        shit = 'A Key'
        shit2 = 'Key'
    elif quantity > 1:
        shit = 'Keys'
        shit2 = 'Keys'
    embed1 = discord.Embed(title=f'> ```💥``` Chug. wait like a sec i\'m generating the keys', description='yes nigger wait', color=discord.Color.orange())
    embed1.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
    embed1.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed1, ephemeral=True)
    timenames = {9999: "Lifetime", 30: "Monthly", 7: "Weekly", 1: "Daily"}  
    timename = timenames[time]


    keys_generated = []

    for _ in range(quantity):
        key = f"mystic" + "-" + "".join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=16))
        with open(f"{dir}/keys.json", "r") as f:
            keys = json.load(f)
        keys[key] = time
        with open(f"{dir}/keys.json", "w") as f:
            json.dump(keys, f, indent=4)
        keys_generated.append(key)

    keys_str = ', '.join(keys_generated)

    embed = discord.Embed(title=f"> ```💥``` Successfully Generated ur keys.", description=f"here:", color=discord.Color.orange())

    embed.add_field(name='Keys:', value=f'{keys_str}', inline=False) 
    embed.add_field(name='Expiration:', value=f'{timename}  ', inline=False)
    embed.set_footer(text='💥 Mystic V6')
    embed.set_thumbnail(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
    await interaction.edit_original_response(embed=embed)


@tree.command(name="redeem", description="|🗝️| Redeem a key to use Mystic API. (Mystic V6)")
async def redeem(interaction: discord.Interaction, key: str, rblxusername: str):
    user_id = interaction.user.id
    member = interaction.user
    user = interaction.user

    with open(f'{dir}/keys.json', 'r') as f:
        keys = json.load(f)
    if key not in keys:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that the key is `incorrect.`', description=f'If you attempt to use a key make sure to use a `correct` one.', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    with open(f"{dir}/users.json", "r") as f:
        users = json.load(f)
    if str(user_id) in users:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have a `Mystic` Subscription.', description=f'If you attempt to use a key while you ALREADY have a subscription ur being dumb.', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')

    guild_id = 1152346539117793350
    guild = client.get_guild(guild_id)
    othercustomerrole = "Bot Customer"
    role_name = "Bot Customer"
    role = discord.utils.get(guild.roles, name=role_name)

    if role:
        await interaction.user.add_roles(role)

        duration = keys[key]
        namesofthelasting = {9999: "Lifetime", 30: "Monthly", 7: "Weekly"}
        currentthing = namesofthelasting[duration]
        expiration = datetime.datetime.now() + datetime.timedelta(days=duration)
        expirationdate = "Never" if duration == 5000 else expiration.strftime("%Y, %B %d, %H:%M")

        del keys[key]
        with open(f"{dir}/keys.json", "w") as f:
            json.dump(keys, f, indent=4)

        try:
            with open(f"{dir}/users.json", "r") as f:
                users = json.load(f)
        except FileNotFoundError:
            users = {}

        users[str(user_id)] = expiration.strftime("%Y-%m-%d-%H:%M:%S")
        with open(f"{dir}/users.json", "w") as f:
            json.dump(users, f, indent=4)
        
        if currentthing == 'Lifetime':
            currentthing = '40 Years worth of '
        elif currentthing == 'Monthly':
            currentthing = '30/31'
        elif currentthing == 'Weekly':
            currentthing = '7'

        embed = discord.Embed(title='> ```💥``` Congratulations. you now have accesss to `mystic` Predictions.', description='The Best Predictor on the market.', color=discord.Color.orange())
        embed.add_field(name='> `⏳` Expiration:', value=f'{currentthing} Days.', inline=False)
        embed.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
        await interaction.response.send_message(embed=embed)
        sigma[str(interaction.user.id)] = {"discordusername": f'{interaction.user}'}
        with open(f"{dir}/ursigma.json", "w") as f:
            json.dump(sigma, f)
        skibiditoilet[str(interaction.user.id)] = {"license": currentthing}
        with open(f"{dir}/urskibidi.json", "w") as f:
            json.dump(skibiditoilet, f)
        ohio[str(interaction.user.id)] = {"rblxusername": f'{rblxusername}'}
        with open(f"{dir}/urohio.json", "w") as f:
            json.dump(ohio, f)
        
                

    else:
        embed = discord.Embed(title="Unexpected error occured.", description=f"notify me chug", color=discord.Color.red())
        await interaction.response.send_message(embed=embed, ephemeral=True)




@tree.command(name="revoke", description="|💥| Spoof someones key using Mystic API. (Mystic V6)")
async def revoke(interaction: discord.Interaction, user: discord.Member):
    member = interaction.user
    if "ticket mods" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you are not a ticket mod of `Mystic`', description=f'If you attempt to generate a key while your not a `ticket mod` it won\'t work 🤷‍♂️', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return


    customerrole = discord.utils.get(user.roles, name='Bot Customer')

    if customerrole:
        await user.remove_roles(customerrole)

        with open(f"{dir}/users.json", "r") as f:
            users = json.load(f)
        if str(user.id) in users:
            del users[str(user.id)]
            with open(f"{dir}/users.json", "w") as f:
                json.dump(users, f, indent=4)

        embed = discord.Embed(
            title=f"> ```💥``` Successfully revoked {user}'s mystic license.",
            color=discord.Color.red())

        embed.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
        await interaction.response.send_message(embed=embed)
        try:
            embed = discord.Embed(title=f'> ```💥``` {member}, Your Access to mystic has been revoked.',
                                description='You no longer have access to `mystic` paid. ask ticket mfs why cuz i have no idea',
                                color=discord.Color.red())
            embed.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
            await user.send(embed=embed)


        except:
            embed = discord.Embed(title="> ```❌``` Your nigger doesn'\t have a license mate.",
                                description=f"Make sure to try and revoke someone with a license nigga",
                                color=discord.Color.red())
            embed.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
            await interaction.response.send_message(embed=embed, ephemeral=True)









@tree.command(name="profile", description="|🤯| Get your profile (Mystic V6)")
async def profilee(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    member = interaction.user
    user = interaction.user
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed2 = discord.Embed(title='> ```💥``` Rendering your profile using `Mystic` API.', description='Wait Patiently.', color=discord.Color.orange())
    embed2.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
    await interaction.response.send_message(embed=embed2)

    user_tokens = auths.get(str(interaction.user.id))
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }

        conn = http.client.HTTPSConnection("api.bloxflip.com")
        conn.request("GET", "/user", "", headers)

        response = conn.getresponse()
        data = json.loads(response.read().decode('utf-8'))

        if "user" in data:
            user_data = data["user"]
            print(user_data)
            user_id = data["user"]["robloxId"]
            username = user_data.get("robloxUsername")
            balance = user_data.get("wallet")
            withdrawn = user_data.get("totalWithdrawn")
            deposited = user_data.get("totalDeposited")
            wagered = user_data.get("wager")
            affcode = user_data.get("affiliateCode")
            affcash = user_data.get("affiliateMoney")
            rakebal = user_data.get("rakebackBalance")
            url = f"https://thumbnails.roblox.com/v1/users/avatar-headshot?userIds={user_id}&size=352x352&format=Png&isCircular=false"
            response = scraper.get(url)
            pfp = response.json()["data"][0]["imageUrl"]

            
            embed = discord.Embed(title=f'> ```💥``` Profile:', description='Current Info about your bloxflip account:', color=discord.Color.orange())
            embed.add_field(name='> `👨‍🦲` ID:', value=f'{user_id}', inline=False)
            embed.add_field(name='> `📛` Username:', value=f'{username}', inline=False)
            embed.add_field(name='> `🤑` Balance:', value=f'{balance}', inline=False)
            embed.add_field(name='> `🏦` Withdrawn:', value=f'{withdrawn}', inline=False)
            embed.add_field(name='> `💵` Deposited:', value=f'{deposited}', inline=False)
            embed.add_field(name='> `🎰` Wagered:', value=f'{wagered}', inline=False)
            embed.add_field(name='> `🤔` Affiliate Code:', value=f'{affcode}', inline=False)
            embed.add_field(name='> `💸` Affiliate Earnings:', value=f'{affcash}', inline=False)
            embed.add_field(name='> `🪙` Rakeback Balance:', value=f'{rakebal}', inline=False)
            embed.set_footer(text='💥 Mystic V6')
            embed.set_thumbnail(url=f'{pfp}')
            embed.set_image(url='https://clipartcraft.com/images/checkmark-clipart-orange.png')
            await interaction.edit_original_response(embed=embed)






import hashlib
import requests

def HashedGenerate(interaction):
    user_tokens = auths.get(str(interaction.user.id))
    
    if user_tokens:
        auth_token = user_tokens.get("auth_token")
        
        if auth_token:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'X-Auth-Token': auth_token
            }

    try:
        conn = http.client.HTTPSConnection("api.bloxflip.com")
        conn.request("GET", f"/games/mines/history?size=100&page=0", "", headers)

        response = conn.getresponse()
            

        if response.status != 200:
            raise Exception(f"HTTP error: {response.status_code}")

        data = json.loads(response.read().decode('utf-8')).get('data', [])
        hashes = [x['serverSeed'] for x in data if not x['exploded']]

        if hashes:

            server_seed = hashes[0][:32]

            conn = http.client.HTTPSConnection("api.bloxflip.com")
            conn.request("POST", f"/provably-fair/clientSeed", {'clientSeed': server_seed}, headers)

            change_hash_req = conn.getresponse()
            
            if change_hash_req.status != 200:
                raise Exception(f"HTTP error: {change_hash_req.status}")

            return json.loads(change_hash_req.read().decode('utf-8')).get('success', False)
        else:
            print("No valid server seeds found.")
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False








@app_commands.choices(method=[
    Choice(name="1. HashedGenerate", value="HashedGenerate"),
])
@tree.command(name='unrig', description='|↪️| Remove rig from your bloxflip account using our API. (Mystic V6)')
async def unrig(interaction: discord.Interaction, method: str):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title='> ```💥``` Unrigging user using Mystic Predictions `API`', description=f'Using HashedGenerate, Please Wait.. ', color=discord.Color.orange())
    embed.set_image(url='https://images-ext-1.discordapp.net/external/JQm-yHOWUZ78S8ZwcO4z3qIg8LxtBrQ-N9wTVHEHSv0/https/clipartcraft.com/images/checkmark-clipart-orange.png?format=webp&quality=lossless&width=478&height=420')
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)

    if method == 'HashedGenerate':
        HashedGenerate(interaction)


    embed1 = discord.Embed(title='> ```💥``` Unrig was a success with Mystic.', description='Check your newly unrigged hash.', color=discord.Color.orange())
    embed1.set_footer(text='💥 Mystic V6.')
    embed1.set_thumbnail(url='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGR0NW1qeTYyZXRjOGk3YTM4NjU1d3ZrZzlxaXpvYmdyejFxOWY1YyZlcD12MV9zdGlja2Vyc19zZWFyY2gmY3Q9cw/iUZfNQLebVcZsJmNPZ/giphy.gif')
    await interaction.edit_original_response(embed=embed1)





Toogled = False

@app_commands.choices(method=[
    Choice(name="1. HashedGenerate", value="HashedGenerate"),
])

@tree.command(name='automaticunrig', description='|🚗| Auto - unrig using mystic API. (Mystic V6)')
async def autounrig(interaction: discord.Interaction, method: str):
    global Toogled
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return

    async def invoke_method(interaction, method):
        if method == 'HashedGenerate':
            HashedGenerate(interaction)


    embed = discord.Embed(title='> ```✅``` Autounrig has been enabled', description='cool', color=discord.Color.green())
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)

    Toogled = True
    while Toogled:
        await invoke_method(interaction, method)
        await asyncio.sleep(1)



@tree.command(name='disableautounrig', description='|🛑| Stops The Unrig Proccess using our API. (Mystic V6)')
async def stopautounrig(interaction: discord.Interaction):
    global Toogled
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    Toogled = False
    embed = discord.Embed(title='> ```❌``` Autounrig has been disabled', description='cool', color=discord.Color.red())
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)


from discord import Embed, Color, Interaction


@tree.command(name='userpredictor_stats', description='|🗽| Obtain information about your time with mystic using mystic API. (Mystic V6)')
async def stats(interaction: Interaction):
    global skibidi
    global totalplayed
    global totalbet
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    
    embed = Embed(title='> ```💥``` Loading your mystic stats.', 
                  description='Please wait while we load stats about you using Mystic API', 
                  color=Color.orange())
    embed.set_footer(text='💥 Mystic V6')
    await interaction.response.send_message(embed=embed)
    
    try:
        totalplayed = lilbro[str(interaction.user.id)]["userpredamount"]
        totalbet = bigbro[str(interaction.user.id)]["userbetamount"]
    except KeyError:
        await interaction.edit_original_response(content="User data not found.", embed=None)
        return
    
    if totalplayed == 1:
        skibidi = 'time'
    else:
        skibidi = 'times'
    
    try:
        embed1 = Embed(title='> ```🪙``` Successfully Loaded Info About You And `Mystic.`',
                       description='Here you go!',
                       color=Color.orange())
        embed1.add_field(name='> `🎯` Total Games Predicted With Mystic:', 
                         value=f'{totalplayed} {skibidi}')
        embed1.add_field(name='> `🅱️` Total Bets Made With Mystic (Towers And Mines):', 
                         value=f'{totalbet}')
        embed1.set_footer(text='💥 Mystic V6')
        await interaction.edit_original_response(embed=embed1)
    except Exception as e:
        await interaction.edit_original_response(content=f"An error occurred: {e}", embed=None)



@tree.command(name='checkuserpaidinfo', description='|🛫| Checks user\'s paid sub. information using our API. (Mystic V6)')
async def checkusershit(interaction: discord.Interaction, userid: str):  
    embed1 = discord.Embed(title='> `🔥` Checking user information..', description='Please wait..', color=discord.Color.orange())   
    embed1.set_footer(text='💥 Mystic V6')  
    await interaction.response.send_message(embed=embed1)
    try:
        niggerr = sigma.get(str(userid))
        if niggerr:
            dcuser = niggerr.get("discordusername")
        kindofanigger = skibiditoilet.get(str(userid))
        if kindofanigger:
            licensetype = kindofanigger.get("license")
        blacknigger = ohio.get(str(userid))
        if blacknigger:
            robloxuser = blacknigger.get("rblxusername")
        embed = discord.Embed(title='> `🛫` Mystic user `checker`', description='Successfully checked user. heres the information:', color=discord.Color.blue())
        embed.add_field(name='> `😎` Discord Username:', value=f'{dcuser}', inline=False)
        embed.add_field(name='> `🥦` License Type:', value=f'{licensetype} Predictor', inline=False)
        embed.add_field(name='> `👤` Roblox Username:', value=f'{robloxuser}', inline=False)
        embed.set_footer(text='💥 Mystic V6')
        await interaction.edit_original_response(embed=embed)
    except Exception as e:
        embed = discord.Embed(title='> `❌` Mystic user `checker` failed', description='failed to check user. heres the information why it might be the case:', color=discord.Color.blue())
        embed.add_field(name='> `💥` Reason 1:', value=f'User never owned our predictor', inline=False)
        embed.add_field(name='> `💥` Reason 2:', value=f'If the user was OG, User did not register their information with /addinfo (Used to reset keys for og mfs)', inline=False)
        embed.add_field(name='> `💥` Reason 3:', value=f'General error in the code (Unlikely)', inline=False)
        embed.add_field(name='> `🛑` Response from the code:', value=f'this error `occured` because of {e}', inline=False)
        embed.set_footer(text='💥 Mystic V6')
        await interaction.edit_original_response(embed=embed)




@app_commands.choices(yourlicensetype=[
    Choice(name="1. Lifetime", value="40 Years worth of"),
    Choice(name="2. Monthly", value="Monthly"),
    Choice(name="3. Weekly", value="Weekly"),
])
@tree.command(name='registerinfo', description='|🎩| Register information about you (FOR OG USERS ONLY) using mystic API. (MYSTIC V6)')
async def skikikiki(interaction: discord.Interaction, yourlicensetype: str, robloxusername: str):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    userid = interaction.user.id
    embed1 = discord.Embed(title='> `🔥` Registering user information..', description='Please wait..', color=discord.Color.orange())   
    embed1.set_footer(text='💥 Mystic V6')  
    await interaction.response.send_message(embed=embed1)
    try:
        sigma[str(interaction.user.id)] = {"discordusername": f'{interaction.user}'}
        with open(f"{dir}/ursigma.json", "w") as f:
            json.dump(sigma, f)
        skibiditoilet[str(interaction.user.id)] = {"license": yourlicensetype}
        with open(f"{dir}/urskibidi.json", "w") as f:
            json.dump(skibiditoilet, f)
        ohio[str(interaction.user.id)] = {"rblxusername": f'{robloxusername}'}
        with open(f"{dir}/urohio.json", "w") as f:
            json.dump(ohio, f)
        

        embed = discord.Embed(title='> `🛫` Mystic user `adder`', description='Successfully added you to the database. heres the information:', color=discord.Color.green())
        embed.add_field(name='> `😎` Discord Username:', value=f'{user}', inline=False)
        embed.add_field(name='> `🥦` License Type:', value=f'{yourlicensetype} Predictor', inline=False)
        embed.add_field(name='> `👤` Roblox Username:', value=f'{robloxusername}', inline=False)
        embed.set_footer(text='💥 Mystic V6')
        await interaction.edit_original_response(embed=embed)
    except Exception as e:
        print(f'{e}')




@tree.command(name='tos', description='|📑| The Terms Of Service of mystic bot. READ THIS BEFORE PURCHASE! (Mystic V6)')
async def tosthingy(interaction: discord.Interaction):
    embed = discord.Embed(title='> `📑` Here is the terms of service of mystic `predictor\'s` paid discord Bot.', description='here:', color=0xffffff)
    embed.add_field(name='> `❔` Section 0.5: (FOR USERS WHO BOUGHT BEFORE 6/15/2024)', value='To reset your mystic predictor bot license key, you must have your information added to the database (/registerinfo to do it)', inline=False)
    embed.add_field(name='> `❔` Section 1:', value='To reset your mystic predictor bot license key, you must have an active license, it must be a Lifetime license though.', inline=False)
    embed.add_field(name='> `❔` Section 2:', value='Do not attempt to scam/lie to us, we will always know.', inline=False)
    embed.add_field(name='> `❔` Section 3:', value='Do not attempt to win the hearts of our staff.', inline=False)
    embed.set_footer(text='🧾 Mystic V6')
    embed.set_thumbnail(url='https://media.giphy.com/media/47uLNY1qXsv6hMOQEU/giphy.gif?cid=790b7611gn4pt91euluko5dpz94im4cynrasmjg4rvd8vsq9&ep=v1_stickers_search&rid=giphy.gif&ct=s')
    await interaction.response.send_message(embed=embed)



@tree.command(name='reportissue', description='|🤓| Report a bug/issue to notris. (Coming soon) (Mystic V6)')
async def tosthingy(interaction: discord.Interaction):
    user = interaction.user
    user_tokens2 = auths.get(str(interaction.user.id))
    if f"{customerrole}" not in [role.name.lower() for role in interaction.user.roles]:
        error_embed = discord.Embed(title=f'> ```❌``` {user}. Our API Detects that you have not purchased `Mystic` Yet.', description=f'If you attempt to use the best paid predictor on the platform, please purchase by opening a `ticket`', color=discord.Color.red())
        error_embed.set_image(url='https://th.bing.com/th/id/R.1bc8311e97b800a409de6d4e915749b1?rik=ZlsHOq7s6U%2fnaA&riu=http%3a%2f%2ficon-library.com%2fimages%2fdeny-icon%2fdeny-icon-3.jpg&ehk=vQQDB3S0WbDiEk6f%2boB%2fJgFIo53ABF9dDiknFCY9jJs%3d&risl=&pid=ImgRaw&r=0')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed, ephemeral=True)
        return
    if not user_tokens2:
        error_embed = discord.Embed(title=f'> `❌` {user}. Our API Detects that you have not Linked to `Mystic` Yet.', description="This issue occured because of the following:", color=discord.Color.red())
        error_embed.add_field(name=f'> `💥` Our `API` Couldn\'t Detect your bloxflip account in our database.', value='Please link your bloxflip account to our database using the `/link` slash command.', inline=False)
        error_embed.set_image(url='https://media.giphy.com/media/l378yQjDMER6TR8Q0/giphy.gif?cid=790b7611nls4hx2znckz0ki7c9bo4hp7umqq1rfg9zrrjkjk&ep=v1_stickers_search&rid=giphy.gif&ct=s')
        error_embed.set_footer(text='💥 Mystic V6')
        await interaction.response.send_message(embed=error_embed)
        return
    embed = discord.Embed(title='> `🤓` This command will be added in the near future.', description='yes.', color=discord.Color.yellow())
    await interaction.response.send_message(embed=embed)

@client.event
async def on_ready():
    synced = await tree.sync()
    print(f"Synced {len(synced)} commands.")
    print(f'We have logged in as {client.user}')


client.run('MTI1NTQyMDY1NjI0MjM5MzE0MQ.GUqOoG.5VCcqvYnPFwp61NjrVfh03QwfDBAfxyAzMtanI')
