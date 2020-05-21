import time
import random
import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.behaviour import OneShotBehaviour
from spade import quit_spade
from spade.message import Message
from spade.template import Template
import numpy as np
#liste des imports propre au reseau de neurones 
import os
import math
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from sklearn.metrics import classification_report 
import h5py
#import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import random
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint 
from tensorflow.keras.models import load_model
from collections import deque
seed = 7
np.random.seed(seed)

finish = False
on_train = True

class IA:
    def __init__(self):
        self.memory  = deque(maxlen=2000)
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        model.add(Dense(6, input_dim=3, activation="relu")) ## on a un seul prix à 
        model.add(Dense(3,activation="linear")) ## une sortie contenant les actions à efecturer : accepter l'offre, decliner l'offre, proposer un autre prix 
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        print("epsilon : {}".format(self.epsilon))
        if np.random.random() < self.epsilon:
            print("random")
            return random.choice(range(0,3))
        print("prediction")
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        losses = []
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            history = self.model.fit(state, target, epochs=1, verbose=0)
            losses.append(history.history["loss"])
        print('------------------ MOYENNE DE LOSS -----------------------')
        print(np.mean(losses))
        print('----------------------------------------------------------')


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

async def train_IA(self):
    global finish
    NB_TRAIN = 500
    msg = "un message"
    print("IA crée")
    await self.wait_msg()
    etats = self.getState() ## obtention de l'etat du jeu 
    
    for i in range(0,NB_TRAIN): ## boucle d'entrainements
        rewards = []
        reward = 0
        step = 0
        while not finish: #boucle de deplacement
            act = self.ia.act(etats)
            print(act)
            reward,msg = self.step(act) # fait avancer l'ia avec l'action chjoisis dans pickAction
            rewards.append(reward)
            etats2 = self.getState() # on recupere tous les etats possibles apres le deplacement
            await self.send_msg(msg)
            await self.wait_msg()
            done = finish

            self.ia.remember(etats, act, reward, etats2, done)
            self.ia.replay()       # internally iterates default (prediction) model
            self.ia.target_train() # iterates target model
            
            etats = etats2
        finish = False
        print("moyenne de reward = {}".format(np.mean(rewards)))
        print("fin de l'entrainement : {}".format(i))  
        self.restart()
        # faire redemarer le vendeur aussi





## AdaptativAgent
     
class Vendeur(Agent):

    class Enchere(CyclicBehaviour):

        async def on_start(self):
            #await self.agent.b.join()
            print("Vendeur : Demarrage de la boucle d'enchère")
            self.concession_vendeur = 0.98
            self.prix_courant = random.choice(range(600,1000))
            self.prix_de_reserve_vendeur = random.choice(range(self.prix_courant-100,self.prix_courant-50))
            self.concession_vendeur = 0.98
            msg = Message(to="buyer@localhost")     # Instantiate the message
            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg.body = str(self.prix_courant)         # Set the message content
            await self.send(msg)
            print("Vendeur : premier message envoyé")

        def restart(self):
            global on_train
            if on_train:
                print("Vendeur : restarting...")
                self.prix_courant = random.choice(range(600,1000))
                self.prix_de_reserve_vendeur = random.choice(range(self.prix_courant-100,self.prix_courant-50))
                self.concession_vendeur = 0.98
            else:
                print("FIN DU VENDEUR")
                self.kill(10)
            
        async def on_end(self):
            print("Vendeur finished with exit code {}.".format(self.exit_code))
            await self.agent.stop()
    
        async def run(self):
            global finish
            msg = await self.receive(timeout=10) # wait for a message for 10 seconds
            if msg:
                if msg.body is "ok":
                    print("Vendeur : Vendu pour {} euros !".format(self.prix_courant))
                    finish = True
                    self.restart()
                else:
                    if msg.body is "ko":
                        print("Vendeur : L'acheteur refuse, fin de la vente!")
                        finish = True
                        self.restart()
                    else:
                        print("Vendeur : Proposition reçu : {} euros".format(msg.body))
                        prix_reçu = float(msg.body)
                        if prix_reçu >= self.prix_de_reserve_vendeur:
                            print("Vendeur : Vendu pour {} euros !".format(prix_reçu))
                            msg = Message(to="buyer@localhost")     # Instantiate the message
                            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
                            msg.body = "ok"
                            finish = True
                            self.restart()
                        
                        else:
                    
                            print("Vendeur : Prix proposé trop bas !")
                            msg = Message(to="buyer@localhost")     # Instantiate the message
                            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
                            if self.prix_courant == self.prix_de_reserve_vendeur:
                                msg.body = "ko"
                                print("Vendeur : Je ne peux pas descendre plus bas que {} euros, je refuse la vente".format(self.prix_de_reserve_vendeur))
                                await self.send(msg)
                                finish = True
                                self.restart()
                                
                            else:
                                self.prix_courant = float(self.prix_courant*self.concession_vendeur)
                                if self.prix_courant < self.prix_de_reserve_vendeur:
                                    self.prix_courant = self.prix_de_reserve_vendeur
                                print("Vendeur : Envoie de la concession de {} euros".format(self.prix_courant))
                                msg.body = str(self.prix_courant)
                                await self.send(msg)
                         
                       
            else:
                print("Vendeur : Did not received any message after 10 seconds")


    async def setup(self):
        
        print("Vendeur : Création du vendeur") 
        self.c = self.Enchere()
        self.add_behaviour(self.c)


class Negociateur(Agent):

    class Enchere(OneShotBehaviour):

        async def on_start(self):
            print("Negociateur : Demarrage de la boucle d'enchère")
            self.proposition_recu = 0
            self.prix_depart_negociateur = random.choice(range(500,900))
            self.argent_totale_negociateur = random.choice(range(self.prix_depart_negociateur+50,self.prix_depart_negociateur+100))
            self.prix_courant_negociateur = self.prix_depart_negociateur
        
        def restart(self):
            print("Negociateur : restarting...")
            self.prix_depart_negociateur = random.choice(range(500,900))
            self.argent_totale_negociateur = random.choice(range(self.prix_depart_negociateur+50,self.prix_depart_negociateur+100))
            self.prix_courant_negociateur = self.prix_depart_negociateur
        
            

        def getState(self):
            return pd.DataFrame({"proposition" : [self.proposition_recu],
                                "argent_courant": [self.prix_courant_negociateur],
                                "argent_totale" : [self.argent_totale_negociateur]
                                }).to_numpy()

        def step(self,act):
            if act == 0: ## accepter l'offre
                msg = "ok"
                if self.proposition_recu < self.argent_totale_negociateur and self.proposition_recu > self.prix_depart_negociateur :
                    reward = self.argent_totale_negociateur-self.proposition_recu
                else:
                    reward = self.argent_totale_negociateur-self.proposition_recu

                
            elif act == 1: ## decliner l'offre 
                msg = "ko"
                if self.prix_courant_negociateur > self.argent_totale_negociateur:
                    reward = self.prix_courant_negociateur-self.argent_totale_negociateur
                else:
                    if self.proposition_recu < self.argent_totale_negociateur and self.proposition_recu > self.prix_depart_negociateur :
                        reward = self.proposition_recu-self.argent_totale_negociateur
                    else : 
                        reward = 0

            elif act == 2: ## fait une proposition
                self.prix_courant_negociateur =  self.prix_courant_negociateur+10
                msg = str(self.prix_courant_negociateur) 

                #if self.proposition_recu < self.argent_totale_negociateur and self.proposition_recu > self.prix_depart_negociateur :
                reward = self.argent_totale_negociateur-self.prix_courant_negociateur
                #else:
                    #reward = self.argent_totale_negociateur-self.proposition_recu 
              

            return reward,msg

        async def wait_msg(self):
            print("Negociateur : Attente d'un message wait_msg")
            msg = await self.receive(timeout=20) # On attend la premiere proposition du vendeur
            if msg:
                if msg.body is "ko" or msg.body is "ko":
                    print("Negociateur ko ou ok , on restart...")
                    if not on_train: # Si on test le modele on arrete à la fin de la premiere vente
                        await self.agent.stop()
                        quit_spade()
                        
                else:
                    print("Negociateur : prix reçu: {}".format(msg.body))
                    prix = float(msg.body)
                    self.proposition_recu = prix
            else:
                print("Negociateur : Did not received any message after 20 seconds")

        async def send_msg(self,data):
            msg = Message(to="seller@localhost")     # Instantiate the message
            msg.set_metadata("performative", "inform")  # Set the "inform" FIPA performative
            msg.body = str(data)
            print("Negociateur : message envoyé : {}".format(data))
            await self.send(msg)

        async def on_end(self):
            global senderagent
            print("Negociateur finished with exit code {}.".format(self.exit_code))
            await self.agent.stop()
            await senderagent.stop()
            quit_spade()
           

        async def run(self):
            global on_train
            global finish
            ##on_train = False # definie si on charge un modele ou on entraine
            on_train = False
            self.ia = IA()
            if on_train:
                await train_IA(self)
                await self.send_msg("ok")
                self.restart()
                self.ia.model.save("best_agent.h5") ## on sauvegarde le meilleur modele
                print("____________FIN DE LENTRAINEMENT__________")
                on_train = False
            else:
                self.ia.model = load_model("best_agent.h5")
                print("Modele chargé")
                print("____________DEBUT DES PREDICTIONS____________")
                finish = False
                await self.wait_msg()
                while not finish:
                    etat = self.getState()
                    reward, msg = self.step(np.argmax(self.ia.model.predict(etat)[0]))
                    await self.send_msg(msg)
                    await self.wait_msg()
                                         
        
    async def setup(self):
        print("Création du négociateur")
        self.c = self.Enchere()
        self.add_behaviour(self.c)

if __name__ == "__main__":
    receiveragent = Negociateur("buyer@localhost", "azerty")
    future = receiveragent.start()
    future.result() # wait for receiver agent to be prepared.
    senderagent = Vendeur("seller@localhost", "azerty")
    future2 = senderagent.start()
    future2.result()

    while receiveragent.is_alive():
        time.sleep(1)

    
    quit_spade()


