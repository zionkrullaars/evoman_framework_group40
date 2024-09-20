################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys
import gzip
import pickle
import numpy
import pygame
from pygame.locals import *
import struct
import evoman.tmx as tmx
from multiprocessing import Process
from multiprocessing.pool import ThreadPool as Pool

from evoman.player import *
from evoman.controller import Controller
from evoman.sensors import Sensors

def checks_params(params):

    # validates parameters values

    if params['multiplemode'] == "yes" and len(params['enemies']) < 2:
        print("ERROR: 'enemies' must contain more than one enemy for multiple mode.")
        sys.exit(0)

    if params['multiplemode'] not in ('yes','no'):
        print("ERROR: 'multiplemode' value must be 'yes' or 'no'.")
        sys.exit(0)

    if params['speed'] not in ('normal','fastest'):
        print("ERROR: 'speed' value must be 'normal' or 'fastest'.")
        sys.exit(0)

    if params['clockprec'] not in ('low','medium'):
        print("ERROR: 'clockprec' value must be 'low' or 'medium'.")
        sys.exit(0)

    if params['sound'] not in ('on','off'):
        print("ERROR: 'sound' value must be 'on' or 'off'.")
        sys.exit(0)

    if type(params['timeexpire']) is not int:
        print("ERROR: 'timeexpire' must be integer.")
        sys.exit(0)

    if type(params['level']) is not int:
        print("ERROR: 'level' must be integer.")
        sys.exit(0)

    if type(params['overturetime']) is not int:
        print("ERROR: 'overturetime' must be integer.")
        sys.exit(0)


    # checks parameters consistency
    if params['multiplemode'] == "no" and len(params['enemies']) > 1:
        print("MESSAGE: there is more than one enemy in 'enemies' list although the mode is not multiple.")

    if params['level'] < 1 or params['level'] > 3:
        print("MESSAGE: 'level' chosen is out of recommended (tested).")

def load_sprites(enemyn, enemyImports, screen, level):

    # loads enemy and map
    if not enemyn in enemyImports:
        enemyImports[enemyn] = __import__('evoman.enemy'+str(enemyn), fromlist=['enemy'+str(enemyn)])
    enemy = enemyImports[enemyn]
    tilemap = tmx.load(enemy.tilemap, screen.get_size())  # map

    sprite_e = tmx.SpriteLayer()
    start_cell = tilemap.layers['triggers'].find('enemy')[0]
    enemy = enemy.Enemy((start_cell.px, start_cell.py), sprite_e, visuals=False)
    tilemap.layers.append(sprite_e)  # enemy

    # loads player
    sprite_p = tmx.SpriteLayer()
    start_cell = tilemap.layers['triggers'].find('player')[0]
    player = Player((start_cell.px, start_cell.py), enemyn, level, sprite_p, visuals =False)
    tilemap.layers.append(sprite_p)

    player.sensors = Sensors()
    enemy.sensors = Sensors()

    return player, enemy, tilemap

def cons_multi(values):
    return values.mean() - values.std()

def play_sep(params,pcont="None"):

    if params['multiplemode'] == "yes":
        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]
        for e in params['enemies']:

            fitness, playerlife, enemylife, time  = run_single_sep(e,pcont,params)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        vfitness = cons_multi(numpy.array(vfitness))
        vplayerlife_new = cons_multi(numpy.array(vplayerlife))
        venemylife_new = cons_multi(numpy.array(venemylife))
        vtime = cons_multi(numpy.array(vtime))

        enemylifesum = numpy.sum(numpy.array(venemylife))
        playerlifesum = numpy.sum(numpy.array(vplayerlife))

        gain = playerlifesum - enemylifesum

        # Count number of enemylife where enemy life == 0
        dead_enemies = numpy.count_nonzero(numpy.array(venemylife) == 0) / len(venemylife)


        return    vfitness, vplayerlife_new, venemylife_new, vtime, gain, dead_enemies
    else:
        fitness, playerlife, enemylife, time = run_single_sep(params['enemyn'],pcont, params)
        gain = playerlife - enemylife
        dead_enemies = 1 if enemylife == 0 else 0
        return fitness, playerlife, enemylife, time, gain, dead_enemies

def runWrapper(env,e,pcont,econt):
    return env.run_single(e,pcont,econt)

def run_single_sep(enemyn, pcont, params):
    """Run a single game simulation.

    Args:
        enemyn: The enemy number.
        pcont: The player controller.
        econt: The enemy controller.
        params: A dictionary containing the necessary parameters from the environment.

    Returns:
        A tuple containing fitness, player life, enemy life, and time.
    """
    # Extract parameters from the dictionary
    clock = pygame.time.Clock()
    flags =  DOUBLEBUF
    screen = pygame.display.set_mode((736, 512), flags)
    clockprec = params['clockprec']
    speed = params['speed']
    playermode = params['playermode']
    sound = params['sound']
    overturetime = params['overturetime']
    visuals = False
    timeexpire = params['timeexpire']
    enemyImports = {e: __import__('evoman.enemy'+str(e), fromlist=['enemy'+str(e)]) for e in params['enemies']}

    def fitness_single(enemylife, playerlife, time):
        return 0.9*(100 - enemylife) + 0.1*playerlife - numpy.log(time)

    # sets controllers
    pcont = pcont

    checks_params(params)

    enemyn = enemyn  # sets the current enemy
    ends = 0
    time = 0
    freeze_p = False
    freeze_e = False
    start = False

    if enemyn not in enemyImports:
        enemyImports[enemyn] = __import__('evoman.enemy' + str(enemyn), fromlist=['enemy' + str(enemyn)])
    enemy = enemyImports[enemyn]

    player, enemy, tilemap = load_sprites(enemyn, enemyImports, screen, params['level'])

    # game main loop
    while True:
        # adjusts frames rate for defining game speed
        if clockprec == "medium":  # medium clock precision
            if speed == 'normal':
                clock.tick_busy_loop(30)
            elif speed == 'fastest':
                clock.tick_busy_loop()
        else:  # low clock precision
            if speed == 'normal':
                clock.tick(30)
            elif speed == 'fastest':
                clock.tick()

        # game timer
        time += 1
        params['time'] = time
        if playermode == "human" or sound == "on":
            # sound effects
            if sound == "on" and time == 1:
                sound = pygame.mixer.Sound('evoman/sounds/open.wav')
                c = pygame.mixer.Channel(1)
                c.set_volume(1)
                c.play(sound, loops=10)

            if time > overturetime:  # delays game start a little bit for human mode
                start = True
        else:
            start = True

        # checks screen closing button
        event = pygame.event.get()
        for e in event:
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return

        # updates objects and draws its items on screen
        tilemap.update(33 / 1000., params)

        if visuals:
            screen.fill((250, 250, 250))
            tilemap.draw(screen)

            # player life bar
            vbar = int(100 * (1 - (player.life / float(player.max_life))))
            pygame.draw.line(screen, (0, 0, 0), [40, 40], [140, 40], 2)
            pygame.draw.line(screen, (0, 0, 0), [40, 45], [140, 45], 5)
            pygame.draw.line(screen, (150, 24, 25), [40, 45], [140 - vbar, 45], 5)
            pygame.draw.line(screen, (0, 0, 0), [40, 49], [140, 49], 2)

            # enemy life bar
            vbar = int(100 * (1 - (enemy.life / float(enemy.max_life))))
            pygame.draw.line(screen, (0, 0, 0), [590, 40], [695, 40], 2)
            pygame.draw.line(screen, (0, 0, 0), [590, 45], [695, 45], 5)
            pygame.draw.line(screen, (194, 118, 55), [590, 45], [695 - vbar, 45], 5)
            pygame.draw.line(screen, (0, 0, 0), [590, 49], [695, 49], 2)

        # gets fitness for training agents
        fitness = fitness_single(enemy.life, player.life, time)

        # returns results of the run
        def return_run():
            return float(fitness), float(player.life), float(enemy.life), int(time)

        if start == False and playermode == "human":
            myfont = pygame.font.SysFont("Comic sams", 100)
            pygame.font.Font.set_bold
            screen.blit(myfont.render("Player", 1, (150, 24, 25)), (50, 180))
            screen.blit(myfont.render("  VS  ", 1, (50, 24, 25)), (250, 180))
            screen.blit(myfont.render("Enemy " + str(enemyn), 1, (194, 118, 55)), (400, 180))

        # checks player life status
        if player.life == 0:
            ends -= 1

            # tells user that player has lost
            if playermode == "human":
                myfont = pygame.font.SysFont("Comic sams", 100)
                pygame.font.Font.set_bold
                screen.blit(myfont.render(" Enemy wins", 1, (194, 118, 55)), (150, 180))

            player.kill()  # removes player sprite
            enemy.kill()  # removes enemy sprite

            if playermode == "human":
                # delays run finalization for human mode
                if ends == -overturetime:
                    return return_run()
            else:
                return return_run()

        # checks enemy life status
        if enemy.life == 0:
            ends -= 1
            if visuals:
                screen.fill((250, 250, 250))
                tilemap.draw(screen)

            # tells user that player has won
            if playermode == "human":
                myfont = pygame.font.SysFont("Comic sams", 100)
                screen.blit(myfont.render(" Player wins ", 1, (150, 24, 25)), (170, 180))

            enemy.kill()  # removes enemy sprite
            player.kill()  # removes player sprite

            if playermode == "human":
                if ends == -overturetime:
                    return return_run()
            else:
                return return_run()

        if params['loadplayer'] == "no":  # removes player sprite from game
            player.kill()

        if params['loadenemy'] == "no":  # removes enemy sprite from game
            enemy.kill()

        # updates screen
        if visuals:
            pygame.display.flip()

        # game runtime limit
        if playermode == 'ai':
            if time >= enemy.timeexpire:
                return return_run()
        else:
            if time >= timeexpire:
                return return_run()

# main class
class Environment(object):


    # simulation parameters
    def __init__(self,
                 experiment_name='test',
                 multiplemode="no",           # yes or no
                 enemies=[1],                 # array with 1 to 8 items, values from 1 to 8
                 loadplayer="yes",            # yes or no
                 loadenemy="yes",             # yes or no
                 level=2,                     # integer
                 playermode="ai",             # ai or human
                 enemymode="static",          # ai or static
                 speed="fastest",             # normal or fastest
                 inputscoded="no",            # yes or no
                 randomini="no",              # yes or no
                 sound="off",                  # on or off
                 contacthurt="player",        # player or enemy
                 logs="on",                   # on or off
                 savelogs="yes",              # yes or no
                 clockprec="low",
                 timeexpire=3000,             # integer
                 overturetime=100,            # integer
                 solutions=None,              # any
                 fullscreen=False,            # True or False
                 player_controller=None,      # controller object
                 enemy_controller=None,      # controller object
                 use_joystick=False,
                 visuals=False):


        # initializes parameters

        self.experiment_name = experiment_name
        self.multiplemode = multiplemode
        self.enemies = enemies
        self.enemyn = enemies[0] # initial current enemy
        self.loadplayer = loadplayer
        self.loadenemy = loadenemy
        self.level = level
        self.playermode = playermode
        self.enemymode = enemymode
        self.speed = speed
        self.inputscoded = inputscoded
        self.randomini = randomini
        self.sound = sound
        self.contacthurt = contacthurt
        self.logs = logs
        self.fullscreen = fullscreen
        self.savelogs = savelogs
        self.clockprec = clockprec
        self.timeexpire = timeexpire
        self.overturetime = overturetime
        self.solutions = solutions
        self.joy = 0
        self.use_joystick = use_joystick

        self.visuals = visuals
        self.enemyImports = {e: __import__('evoman.enemy'+str(e), fromlist=['enemy'+str(e)]) for e in self.enemies}



        # initializes default random controllers

        if self.playermode == "ai" and player_controller == None:
            self.player_controller = Controller()
        else:
            self.player_controller =  player_controller

        if self.enemymode == "ai" and enemy_controller == None:
            self.enemy_controller = Controller()
        else:
            self.enemy_controller =  enemy_controller


        # initializes log file
        if self.logs  == "on" and self.savelogs == "yes":
            file_aux  = open(self.experiment_name+'/evoman_logs.txt','w')
            file_aux.close()


        # initializes pygame library
        pygame.init()
        self.print_logs("MESSAGE: Pygame initialized for simulation.")

        # initializes sound library for playing mode
        if self.sound == "on":
            pygame.mixer.init()
            self.print_logs("MESSAGE: sound has been turned on.")

        # initializes joystick library
        if self.use_joystick:
            pygame.joystick.init()
            self.joy = pygame.joystick.get_count()

        self.clock = pygame.time.Clock() # initializes game clock resource
        
        if self.fullscreen:
            flags =  DOUBLEBUF  |  FULLSCREEN
        else:
            flags =  DOUBLEBUF

        self.screen = pygame.display.set_mode((736, 512), flags)

        self.screen.set_alpha(None) # disables uneeded alpha
        pygame.event.set_allowed([QUIT, KEYDOWN, KEYUP]) # enables only needed events


        self.load_sprites()



    def load_sprites(self):

        # loads enemy and map
        if not self.enemyn in self.enemyImports:
            self.enemyImports[self.enemyn] = __import__('evoman.enemy'+str(self.enemyn), fromlist=['enemy'+str(self.enemyn)])
        enemy = self.enemyImports[self.enemyn]
        self.tilemap = tmx.load(enemy.tilemap, self.screen.get_size())  # map

        self.sprite_e = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('enemy')[0]
        self.enemy = enemy.Enemy((start_cell.px, start_cell.py), self.sprite_e, visuals=self.visuals)
        self.tilemap.layers.append(self.sprite_e)  # enemy

        # loads player
        self.sprite_p = tmx.SpriteLayer()
        start_cell = self.tilemap.layers['triggers'].find('player')[0]
        self.player = Player((start_cell.px, start_cell.py), self.enemyn, self.level, self.sprite_p, visuals =self.visuals)
        self.tilemap.layers.append(self.sprite_p)

        self.player.sensors = Sensors()
        self.enemy.sensors = Sensors()


    # updates environment with backup of current solutions in simulation
    def get_solutions(self):
        return self.solutions


        # method for updating solutions bkp in simulation
    def update_solutions(self, solutions):
        self.solutions = solutions


    # method for updating simulation parameters
    def update_parameter(self, name, value):

        if type(value) is str:
            exec('self.'+name +"= '"+ value+"'")
        else:
            exec('self.'+name +"= "+ str(value))

        self.print_logs("PARAMETER CHANGE: "+name+" = "+str(value))



    def print_logs(self, msg):
        if self.logs == "on":
            print('\n'+msg) # prints log messages to screen

            if self.savelogs == "yes": # prints log messages to file
                file_aux  = open(self.experiment_name+'/evoman_logs.txt','a')
                file_aux.write('\n\n'+msg)
                file_aux.close()


    def get_num_sensors(self):

        if hasattr(self, 'enemy') and self.enemymode == "ai":
            return  len(self.enemy.sensors.get(self))
        else:
            if hasattr(self, 'player') and self.playermode == "ai":
                return len(self.player.sensors.get(self))
            else:
                return 0


    # writes all variables related to game state into log
    def state_to_log(self):


        self.print_logs("########## Simulation state - INI ###########")
        if self.solutions == None:
            self.print_logs("# solutions # : EMPTY ")
        else:
            self.print_logs("# solutions # : LOADED ")

        self.print_logs("# sensors # : "+ str( self.get_num_sensors() ))
        self.print_logs(" ------  parameters ------  ")
        self.print_logs("# contact hurt (training agent) # : "  +self.contacthurt)

        self.print_logs("multiple mode: "+self.multiplemode)

        en = ''
        for e in self.enemies:
            en += ' '+str(e)
        self.print_logs("enemies list:"+ en)

        self.print_logs("current enemy: " +str(self.enemyn))
        self.print_logs("player mode: " +self.playermode)
        self.print_logs("enemy mode: "  +self.enemymode)
        self.print_logs("level: " +str(self.level))
        self.print_logs("clock precision: "+ self.clockprec)
        self.print_logs("inputs coded: "  +self.inputscoded)
        self.print_logs("random initialization: "  +self.randomini)
        self.print_logs("expiration time: "  +str(self.timeexpire))
        self.print_logs("speed: " +self.speed)
        self.print_logs("load player: " +self.loadplayer)
        self.print_logs("load enemy: " +self.loadenemy)
        self.print_logs("sound: "  +self.sound)
        self.print_logs("overture time: "  +str(self.overturetime))
        self.print_logs("logs: "+self.logs)
        self.print_logs("save logs: "+self.savelogs)
        self.print_logs("########## Simulation state - END ###########")



    # exports current environment state to files
    def save_state(self):

        # saves configuration file for simulation parameters
        file_aux  = open(self.experiment_name+'/evoman_paramstate.txt','w')
        en = ''
        for e in self.enemies:
            en += ' '+str(e)
        file_aux.write("\nenemies"+ en)
        file_aux.write("\ntimeexpire "  +str(self.timeexpire))
        file_aux.write("\nlevel " +str(self.level))
        file_aux.write("\nenemyn " +str(self.enemyn))
        file_aux.write("\noverturetime "  +str(self.overturetime))
        file_aux.write("\nplayermode " +self.playermode)
        file_aux.write("\nenemymode "  +self.enemymode)
        file_aux.write("\ncontacthurt "  +self.contacthurt)
        file_aux.write("\nclockprec "+ self.clockprec)
        file_aux.write("\ninputscoded "  +self.inputscoded)
        file_aux.write("\nrandomini "  +self.randomini)
        file_aux.write("\nmultiplemode "+self.multiplemode)
        file_aux.write("\nspeed " +self.speed)
        file_aux.write("\nloadplayer " +self.loadplayer)
        file_aux.write("\nloadenemy " +self.loadenemy)
        file_aux.write("\nsound "  +self.sound)
        file_aux.write("\nlogs "+self.logs)
        file_aux.write("\nsavelogs "+self.savelogs)
        file_aux.close()

        # saves state of solutions in the simulation
        file = gzip.open(self.experiment_name+'/evoman_solstate', 'w', compresslevel = 5)
        pickle.dump(self.solutions, file, protocol=2)
        file.close()


        self.print_logs("MESSAGE: state has been saved to files.")



    # loads a state for environment from files
    def load_state(self):


        try:

            # loads parameters
            state = open(self.experiment_name+'/evoman_paramstate.txt','r')
            state = state.readlines()
            for idp,p in enumerate(state):
                pv = p.split(' ')

                if idp>0:    # ignore first line
                    if idp==1: # enemy list
                        en = []
                        for i in range(1,len(pv)):
                            en.append(int(pv[i].rstrip('\n')))
                        self.update_parameter(pv[0], en)
                    elif idp<6: # numeric params
                        self.update_parameter(pv[0], int(pv[1].rstrip('\n')))
                    else: # string params
                        self.update_parameter(pv[0], pv[1].rstrip('\n'))

            # loads solutions
            file = gzip.open(self.experiment_name+'/evoman_solstate')
            self.solutions =  pickle.load(file, encoding='latin1')
            self.print_logs("MESSAGE: state has been loaded.")

        except IOError:
            self.print_logs("ERROR: could not load state.")




    def checks_params(self):

        # validates parameters values

        if self.multiplemode == "yes" and len(self.enemies) < 2:
            self.print_logs("ERROR: 'enemies' must contain more than one enemy for multiple mode.")
            sys.exit(0)

        if self.enemymode not in ('static','ai'):
            self.print_logs("ERROR: 'enemy mode' must be 'static' or 'ai'.")
            sys.exit(0)

        if self.playermode not in ('human','ai'):
            self.print_logs("ERROR: 'player mode' must be 'human' or 'ai'.")
            sys.exit(0)

        if self.loadplayer not in ('yes','no'):
            self.print_logs("ERROR: 'load player' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.loadenemy not in ('yes','no'):
            self.print_logs("ERROR: 'load enemy' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.inputscoded not in ('yes','no'):
            self.print_logs("ERROR: 'inputs coded' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.multiplemode not in ('yes','no'):
            self.print_logs("ERROR: 'multiplemode' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.randomini not in ('yes','no'):
            self.print_logs("ERROR: 'random ini' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.savelogs not in ('yes','no'):
            self.print_logs("ERROR: 'save logs' value must be 'yes' or 'no'.")
            sys.exit(0)

        if self.speed not in ('normal','fastest'):
            self.print_logs("ERROR: 'speed' value must be 'normal' or 'fastest'.")
            sys.exit(0)

        if self.logs not in ('on','off'):
            self.print_logs("ERROR: 'logs' value must be 'on' or 'off'.")
            sys.exit(0)

        if self.clockprec not in ('low','medium'):
            self.print_logs("ERROR: 'clockprec' value must be 'low' or 'medium'.")
            sys.exit(0)

        if self.sound not in ('on','off'):
            self.print_logs("ERROR: 'sound' value must be 'on' or 'off'.")
            sys.exit(0)

        if self.contacthurt not in ('player','enemy'):
            self.print_logs("ERROR: 'contacthurt' value must be 'player' or 'enemy'.")
            sys.exit(0)

        if type(self.timeexpire) is not int:
            self.print_logs("ERROR: 'timeexpire' must be integer.")
            sys.exit(0)

        if type(self.level) is not int:
            self.print_logs("ERROR: 'level' must be integer.")
            sys.exit(0)

        if type(self.overturetime) is not int:
            self.print_logs("ERROR: 'overturetime' must be integer.")
            sys.exit(0)


        # checks parameters consistency

        if self.multiplemode == "no" and len(self.enemies) > 1:
            self.print_logs("MESSAGE: there is more than one enemy in 'enemies' list although the mode is not multiple.")

        if self.level < 1 or self.level > 3:
            self.print_logs("MESSAGE: 'level' chosen is out of recommended (tested).")




            # default fitness function for single solutions
    def fitness_single(self):
        return 0.9*(100 - self.get_enemylife()) + 0.1*self.get_playerlife() - numpy.log(self.get_time())

    # default fitness function for consolidating solutions among multiple games
    def cons_multi(self,values):
        return values.mean() - values.std()

    # measures the energy of the player
    def get_playerlife(self):
        return self.player.life

    # measures the energy of the enemy
    def get_enemylife(self):
        return self.enemy.life

    # gets run time
    def get_time(self):
        return self.time


    # runs game for a single enemy
    def run_single(self,enemyn,pcont,econt):

        # sets controllers
        self.pcont = pcont
        self.econt = econt

        self.checks_params()


        self.enemyn = enemyn # sets the current enemy
        ends = 0
        self.time = 0
        self.freeze_p = False
        self.freeze_e = False
        self.start = False

        if not self.enemyn in self.enemyImports:
            self.enemyImports[self.enemyn] = __import__('evoman.enemy'+str(self.enemyn), fromlist=['enemy'+str(self.enemyn)])
        enemy = self.enemyImports[self.enemyn]

        self.load_sprites()


        # game main loop

        while 1:

            # adjusts frames rate for defining game speed

            if self.clockprec == "medium":  # medium clock precision
                if self.speed == 'normal':
                    self.clock.tick_busy_loop(30)
                elif self.speed == 'fastest':
                    self.clock.tick_busy_loop()

            else:   # low clock precision

                if self.speed == 'normal':
                    self.clock.tick(30)
                elif self.speed == 'fastest':
                    self.clock.tick()


            # game timer
            self.time += 1
            if self.playermode == "human" or self.sound == "on":
                # sound effects
                if self.sound == "on" and self.time == 1:
                    sound = pygame.mixer.Sound('evoman/sounds/open.wav')
                    c = pygame.mixer.Channel(1)
                    c.set_volume(1)
                    c.play(sound,loops=10)

                if self.time > self.overturetime: # delays game start a little bit for human mode
                    self.start = True
            else:
                self.start = True


            # checks screen closing button
            self.event = pygame.event.get()
            for event in  self.event:
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            # updates objects and draws its itens on screen
            self.tilemap.update( 33 / 1000., self)

            if self.visuals:
                
                self.screen.fill((250,250,250))
                self.tilemap.draw(self.screen)

                # player life bar
                vbar = int(100 *( 1-(self.player.life/float(self.player.max_life)) ))
                pygame.draw.line(self.screen, (0,   0,   0), [40, 40],[140, 40], 2)
                pygame.draw.line(self.screen, (0,   0,   0), [40, 45],[140, 45], 5)
                pygame.draw.line(self.screen, (150,24,25),   [40, 45],[140 - vbar, 45], 5)
                pygame.draw.line(self.screen, (0,   0,   0), [40, 49],[140, 49], 2)

                # enemy life bar
                vbar = int(100 *( 1-(self.enemy.life/float(self.enemy.max_life)) ))
                pygame.draw.line(self.screen, (0,   0,   0), [590, 40],[695, 40], 2)
                pygame.draw.line(self.screen, (0,   0,   0), [590, 45],[695, 45], 5)
                pygame.draw.line(self.screen, (194,118,55),  [590, 45],[695 - vbar, 45], 5)
                pygame.draw.line(self.screen, (0,   0,   0), [590, 49],[695, 49], 2)


            #gets fitness for training agents
            fitness = self.fitness_single()


            # returns results of the run
            def return_run():
                #self.print_logs("RUN: run status: enemy: "+str(self.enemyn)+"; fitness: " + str(fitness) + "; player life: " + str(self.player.life)  + "; enemy life: " + str(self.enemy.life) + "; time: " + str(self.time))
                return  float(fitness), float(self.player.life), float(self.enemy.life), int(self.time)



            if self.start == False and self.playermode == "human":

                myfont = pygame.font.SysFont("Comic sams", 100)
                pygame.font.Font.set_bold
                self.screen.blit(myfont.render("Player", 1,  (150,24,25)), (50, 180))
                self.screen.blit(myfont.render("  VS  ", 1,  (50,24,25)), (250, 180))
                self.screen.blit(myfont.render("Enemy "+str(self.enemyn), 1,  (194,118,55)), (400, 180))


            # checks player life status
            if self.player.life == 0:
                ends -= 1

                # tells user that player has lost
                if self.playermode == "human":
                    myfont = pygame.font.SysFont("Comic sams", 100)
                    pygame.font.Font.set_bold
                    self.screen.blit(myfont.render(" Enemy wins", 1, (194,118,55)), (150, 180))

                self.player.kill() # removes player sprite
                self.enemy.kill()  # removes enemy sprite

                if self.playermode == "human":
                    # delays run finalization for human mode
                    if ends == -self.overturetime:
                        return return_run()
                else:
                    return return_run()


            # checks enemy life status
            if self.enemy.life == 0:
                ends -= 1
                if self.visuals:
                    self.screen.fill((250,250,250))
                    self.tilemap.draw(self.screen)

                # tells user that player has won
                if self.playermode == "human":
                    myfont = pygame.font.SysFont("Comic sams", 100)

                    self.screen.blit(myfont.render(" Player wins ", 1, (150,24,25) ), (170, 180))

                self.enemy.kill()   # removes enemy sprite
                self.player.kill()  # removes player sprite

                if self.playermode == "human":
                    if ends == -self.overturetime:
                        return return_run()
                else:
                    return return_run()


            if self.loadplayer == "no":# removes player sprite from game
                self.player.kill()

            if self.loadenemy == "no":  #removes enemy sprite from game
                self.enemy.kill()

                # updates screen
            if self.visuals:
                pygame.display.flip()


            # game runtime limit
            if self.playermode == 'ai':
                if self.time >= enemy.timeexpire:
                    return return_run()

            else:
                if self.time >= self.timeexpire:
                    return return_run()



    # repeats run for every enemy in list
    def multiple(self,pcont,econt):

        vfitness, vplayerlife, venemylife, vtime = [],[],[],[]

        # with Pool(processes=8) as pool:
        #     # Use starmap
        #     results = pool.starmap(self.run_single, [(e,pcont,econt) for e in self.enemies])
        #     for res in results:
        #         vfitness.append(res[0])
        #         vplayerlife.append(res[1])
        #         venemylife.append(res[2])
        #         vtime.append(res[3])
        for e in self.enemies:

            fitness, playerlife, enemylife, time  = self.run_single(e,pcont,econt)
            vfitness.append(fitness)
            vplayerlife.append(playerlife)
            venemylife.append(enemylife)
            vtime.append(time)

        vfitness = self.cons_multi(numpy.array(vfitness))
        vplayerlife_new = self.cons_multi(numpy.array(vplayerlife))
        venemylife_new = self.cons_multi(numpy.array(venemylife))
        vtime = self.cons_multi(numpy.array(vtime))

        enemylifesum = numpy.sum(numpy.array(venemylife))
        playerlifesum = numpy.sum(numpy.array(vplayerlife))

        gain = playerlifesum - enemylifesum

        # Count number of enemylife where enemy life == 0
        dead_enemies = numpy.count_nonzero(numpy.array(venemylife) == 0) / len(venemylife)


        return    vfitness, vplayerlife_new, venemylife_new, vtime, gain, dead_enemies


    # checks objective mode
    def play(self,pcont="None",econt="None"):

        if self.multiplemode == "yes":
            return self.multiple(pcont,econt)
        else:
            fitness, playerlife, enemylife, time = self.run_single(self.enemies[0],pcont,econt)
            gain = playerlife - enemylife
            dead_enemies = 1 if enemylife == 0 else 0
            return fitness, playerlife, enemylife, time, gain, dead_enemies
