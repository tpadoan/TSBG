import PySimpleGUI as sg
from PIL import Image, ImageTk
import random
import sys

from logic import Game


class ScotlandYardGUI:
    def __init__(self):
        self.window = None
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        self.marco_last_loc = None
        self.detective_starting_loc = None

    def create_window(self):
        window_width = int(self.screen_width * 0.7)
        window_height = int(self.screen_height * 0.4)

        marco_img = Image.open('./img/marco.gif')
        alpha_marco = marco_img.size[0]/marco_img.size[1]
        img_size_marco = (int(0.7*window_height*alpha_marco), int(0.7*window_height))
        marco_img = marco_img.resize(img_size_marco)

        mappa_img = Image.open('./img/ts_map.jpg')
        alpha_mappa = mappa_img.size[0]/mappa_img.size[1]
        img_size_mappa = (int(0.7*window_height*alpha_mappa), int(0.7*window_height))
        mappa_img = mappa_img.resize(img_size_mappa)

        win_img = Image.open('./img/thug_life.jpg')
        alpha_win = win_img.size[0]/win_img.size[1]
        img_size_win = (int(0.7*window_height*alpha_win), int(0.7*window_height))
        win_img = win_img.resize(img_size_win)

        lose_img = Image.open('./img/captured.jpg')
        alpha_lose = lose_img.size[0]/lose_img.size[1]
        img_size_lose = (int(0.7*window_height*alpha_lose), int(0.7*window_height))
        lose_img = lose_img.resize(img_size_lose)

        marco_img_data = ImageTk.PhotoImage(marco_img)
        mappa_img_data = ImageTk.PhotoImage(mappa_img)
        win_img_data = ImageTk.PhotoImage(win_img)
        lose_img_data = ImageTk.PhotoImage(lose_img)

        start_txt ="Il pinguino Marco deve scappare dai dipendenti dell'acquario di Trieste\nper ottenere pesce extra dai passanti. Può spostarsi nella città con la bicicletta,\nl'autobus, o il delfino verde. Scegli le tue mosse per non farti acchiappare!"
        initial_layout = [
            [sg.Text(start_txt, size=(img_size_marco[0], 4), font='Tahoma 13 bold',  key='-GAME_DESCRIPTION-')],
            [sg.Image(key='-MARCO-', size = img_size_marco)],
            [sg.Button("Inizia a giocare!", key='-START-', font='Tahoma 13 bold', size=(20, 5))]
        ]

        pre_game_layout = [
            [sg.Image(key='-MAPPA_PREGAME-', size=img_size_mappa)],
            [sg.Text('', font='Tahoma 13 bold', key='-D1_START_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold',  key='-D2_START_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold',  key='-D3_START_LOCATION-')],
            [sg.Text('Inserisci la posizione iniziale di Marco:', font='Tahoma 13 bold', key='-MARCO_LOCATION-'),
             sg.Input('', enable_events=True, key='-MARCO_LOC_INPUT-', font=('Arial Bold', 20), justification='left', size=(5,1))],
            [sg.Button('Ok', key='-OK_POSITION-', font='Tahoma 13 bold', size=(10, 2))]
        ]

        game_layout = [
            [sg.Image(key='-MAPPA-', size=img_size_mappa), sg.Text("Turn 1", key='-COUNTER-', font='Tahoma 13 bold', justification='right', size=(10,1))],
            [sg.Text('', font='Tahoma 13 bold',  key='-D1_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-D2_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-D3_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-MARCO_LOCATION_TXT-')],
            [sg.Button('Bicicletta', key='-BICI-', button_color='deep sky blue', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('Autobus', key='-BUS-', button_color='red', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('Delfino verde', key='-DELFINO-', button_color='green3', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('CATTURATO!', key='-CATTURATO-', font='Tahoma 13 bold', size=(20, 5))] 
        ]

        win_txt = "Congratulazioni, sei riuscito a fuggire! Ti sei meritato un sacco di pesce!"

        win_layout = [
            [sg.Text(win_txt, size=(img_size_marco[0], 4), font='Tahoma 13 bold',  key='-WIN_DESCRIPTION-')],
            [sg.Image(key='-WIN_IMG-', size = img_size_win)],
            [sg.Button("Ricomincia", key='-WIN_RESTART-', font='Tahoma 13 bold', size=(20, 5))]
        ]

        lose_txt = "Oh no, sei stato catturato! Si ritorna all'acquario!"

        lose_layout = [
            [sg.Text(lose_txt, size=(img_size_marco[0], 4), font='Tahoma 13 bold',  key='-LOSE_DESCRIPTION-')],
            [sg.Image(key='-LOSE_IMG-', size = img_size_lose)],
            [sg.Button("Ricomincia", key='-LOSE_RESTART-', font='Tahoma 13 bold', size=(20, 5))]
        ]

        layout = [[sg.Column(initial_layout, key='-IN_LAYOUT-'), 
                  sg.Column(game_layout, visible=False, key='-GAME_LAYOUT-'),
                  sg.Column(pre_game_layout, visible=False, key='-PRE_GAME_LAYOUT-'),
                  sg.Column(win_layout, visible=False, key='-WIN_LAYOUT-'),
                  sg.Column(lose_layout, visible=False, key='-LOSE_LAYOUT-')]
                  ]
        self.window = sg.Window("La fuga di Marco", layout, size=(window_width, window_height), element_justification='c', finalize=True)
        self.window['-MARCO-'].update(data=marco_img_data)
        self.window['-MAPPA-'].update(data=mappa_img_data)
        self.window['-MAPPA_PREGAME-'].update(data=mappa_img_data)
        self.window['-WIN_IMG-'].update(data=win_img_data)
        self.window['-LOSE_IMG-'].update(data=lose_img_data)

    def switch_to_pre_game_layout(self):
        self.window['-IN_LAYOUT-'].update(visible=False)
        self.window['-PRE_GAME_LAYOUT-'].update(visible=True)

    def switch_to_game_layout(self, marco_starting_pos):
        self.window['-MARCO_LOCATION_TXT-'].update('Posizione iniziale di Marco: '+ str(marco_starting_pos))
        self.window['-PRE_GAME_LAYOUT-'].update(visible=False)
        self.window['-GAME_LAYOUT-'].update(visible=True)

    def switch_to_endgame(self, user_win):
        if user_win:
            self.window['-GAME_LAYOUT-'].update(visible=False)
            self.window['-WIN_LAYOUT-'].update(visible=True)
        else:
            self.window['-GAME_LAYOUT-'].update(visible=False)
            self.window['-LOSE_LAYOUT-'].update(visible=True)


    def set_detective_starting_loc(self, detective_loc):
        self.window['-D1_START_LOCATION-'].update('Detective 1: '+ str(detective_loc[0]))
        self.window['-D2_START_LOCATION-'].update('Detective 2: '+ str(detective_loc[1]))
        self.window['-D3_START_LOCATION-'].update('Detective 3: '+ str(detective_loc[2]))
        self.window['-D1_LOCATION-'].update('Detective 1: '+ str(detective_loc[0]))
        self.window['-D2_LOCATION-'].update('Detective 2: '+ str(detective_loc[1]))  
        self.window['-D3_LOCATION-'].update('Detective 3: '+ str(detective_loc[2]))
        self.detective_starting_loc=detective_loc

    def update_detective(self, detective_loc):
        self.window['-D1_LOCATION-'].update('Detective 1: '+ str(detective_loc[0]))
        self.window['-D2_LOCATION-'].update('Detective 2: '+ str(detective_loc[1]))
        self.window['-D3_LOCATION-'].update('Detective 3: '+ str(detective_loc[2]))

    def update_counter(self, counter):
        self.window['-COUNTER-'].update('Turn ' + str(counter))  

    def restart_layout(self, win):
        if win:
            self.window['-WIN_LAYOUT-'].update(visible=False)    
        else:
            self.window['-LOSE_LAYOUT-'].update(visible=False)
        self.window['-PRE_GAME_LAYOUT-'].update(visible=True)                 

if __name__ == "__main__":
    game_gui = ScotlandYardGUI()
    game_gui.create_window()
    counter = 0
    game = None
    event2move_dict = {'-BICI-':'cart', '-BUS-':'tram', '-DELFINO-':'boat'}
    while True:
        event, values = game_gui.window.read()

        if event == sg.WINDOW_CLOSED:
            break

        elif event == '-START-':
            detective_loc = random.sample(range(1, 22), 3)
            game_gui.set_detective_starting_loc(detective_loc)
            game_gui.switch_to_pre_game_layout()

        elif event == '-OK_POSITION-':
            pos = ''
            for x in values['-MARCO_LOC_INPUT-']:
                pos += x
            mrx_starting_loc = int(pos)
            if mrx_starting_loc not in range(1,22):
                sg.popup("Posizione iniziale non valida!")
                game_gui.window['-MARCO_LOC_INPUT-'].update(values['-MARCO_LOC_INPUT-'][:-1])
            else:
                game = Game()
                game.initGame(detectives=game_gui.detective_starting_loc, mrX=mrx_starting_loc)
                game_gui.switch_to_game_layout(mrx_starting_loc)


        elif event in ['-BICI-', '-BUS-', '-DELFINO-']:
            if counter >= 9:
                game_gui.switch_to_endgame(user_win=True)
            move = event2move_dict[event]
            new_detective_loc = game.playTurn(move)
            game_gui.update_detective(new_detective_loc)
            counter += 1
            game_gui.update_counter(counter=counter+1)

        elif event == '-CATTURATO-':
            game_gui.switch_to_endgame(user_win=False)

        elif event == '-WIN_RESTART-':
            detective_loc = random.sample(range(1, 22), 3)
            counter=0
            game_gui.set_detective_starting_loc(detective_loc)
            game_gui.restart_layout(win=True)
        elif event == '-LOSE_RESTART-':
            detective_loc = random.sample(range(1, 22), 3)
            counter=0
            game_gui.set_detective_starting_loc(detective_loc)
            game_gui.restart_layout(win=False)

    game_gui.window.close()