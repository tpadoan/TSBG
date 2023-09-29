import PySimpleGUI as sg
from PIL import Image, ImageTk
import random
from io import BytesIO
from logic import Game

class ScotlandYardGUI:
    def __init__(self):
        self.window = None
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        self.marco_last_loc = None
        self.detective_starting_loc = None
        self.w_factor = self.screen_width / 2560
        self.h_factor = self.screen_height / 1440
        self.loc_list=[
                    (225, 100), 
                    (310, 190), 
                    (412, 205), 
                    (515, 295), 
                    (555, 425), 
                    (505, 130), 
                    (650, 285), 
                    (675, 457), 
                    (563, 530), 
                    (435, 733), 
                    (635, 80), 
                    (730, 335), 
                    (735, 580), 
                    (800, 205), 
                    (825, 298), 
                    (875, 487), 
                    (710, 737),
                    (850, 60),
                    (940, 220),
                    (1015, 550),
                    (940, 675)
        ]
        for i in range(len(self.loc_list)):
          self.loc_list[i] = (int(self.w_factor*self.loc_list[i][0]), int(self.h_factor*self.loc_list[i][1]))
        self.det_name=['A', 'B', 'C']

    def create_window(self):
        window_width = int(self.screen_width * 0.8)
        window_height = int(self.screen_height * 0.8)

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
        win_img_data = ImageTk.PhotoImage(win_img)
        lose_img_data = ImageTk.PhotoImage(lose_img)
        #mappa_img_data = ImageTk.PhotoImage(mappa_img)
        with BytesIO() as output:
            mappa_img.save(output, format="PNG")
            mappa_img_data = output.getvalue()

        start_txt ="Il pinguino Marco deve scappare dai dipendenti dell'acquario di Trieste\nper ottenere pesce extra dai passanti. Può spostarsi nella città con la bicicletta,\nl'autobus, o il traghetto. Scegli le tue mosse per non farti acchiappare!"
        initial_layout = [
            [sg.Text(start_txt, size=(img_size_marco[0], 4), font='Tahoma 13 bold',  key='-GAME_DESCRIPTION-')],
            [sg.Image(key='-MARCO-', size = img_size_marco)],
            [sg.Button("Inizia a giocare!", key='-START-', font='Tahoma 13 bold', size=(20, 5))]
        ]

        pre_game_layout = [
            [sg.Graph(
                canvas_size=img_size_mappa,
                graph_bottom_left=(0, 0),
                graph_top_right=img_size_mappa,
                key="-PRE_GRAPH-"
            )],
            [sg.Text('', font='Tahoma 13 bold', key='-D1_START_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold',  key='-D2_START_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold',  key='-D3_START_LOCATION-')],
            [sg.Text('Inserisci la posizione iniziale di Marco:', font='Tahoma 13 bold', key='-MARCO_LOCATION-'),
             sg.Input('', enable_events=True, key='-MARCO_LOC_INPUT-', font=('Arial Bold', 20), justification='left', size=(5,1))],
            [sg.Button('Ok', key='-OK_POSITION-', font='Tahoma 13 bold', size=(10, 2))]
        ]

        game_layout = [
            [sg.Graph(
                canvas_size=img_size_mappa,
                graph_bottom_left=(0, 0),
                graph_top_right=img_size_mappa,
                key="-GAME_GRAPH-"
            ), sg.Text("Turno 1", key='-COUNTER-', font='Tahoma 13 bold', justification='left', size=(25,3))],
            [sg.Text('', font='Tahoma 13 bold',  key='-D1_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-D2_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-D3_LOCATION-')],
            [sg.Text('', font='Tahoma 13 bold', key='-MARCO_LOCATION_TXT-')],
            [sg.Button('Bicicletta', key='-BICI-', button_color='deep sky blue', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('Autobus', key='-BUS-', button_color='red', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('Traghetto', key='-DELFINO-', button_color='green3', font='Tahoma 13 bold', size=(20, 5)),
             sg.Button('CATTURATO!', key='-CATTURATO-', font='Tahoma 13 bold', size=(20, 5))] 
        ]

        win_txt = "Congratulazioni, sei riuscito a fuggire! Ti sei meritato un sacco di pesce!"

        win_layout = [
            [sg.Text(win_txt, size=(img_size_marco[0], 3), font='Tahoma 13 bold',  key='-WIN_DESCRIPTION-')],
            [sg.Image(key='-WIN_IMG-', size = img_size_win)],
            [sg.Button("Ricomincia", key='-WIN_RESTART-', font='Tahoma 13 bold', size=(20, 5))]
        ]

        lose_txt = "Oh no, sei stato catturato! Si ritorna all'acquario!"

        lose_layout = [
            [sg.Text(lose_txt, size=(img_size_marco[0], 1), font='Tahoma 13 bold',  key='-LOSE_DESCRIPTION-')],
            [sg.Text('', size=(img_size_marco[0], 1), font='Tahoma 13 bold',  key='-D1_PATH-')],
            [sg.Text('', size=(img_size_marco[0], 1), font='Tahoma 13 bold',  key='-D2_PATH-')],
            [sg.Text('', size=(img_size_marco[0], 1), font='Tahoma 13 bold',  key='-D3_PATH-')],
            [sg.Text('', size=(img_size_marco[0], 1), font='Tahoma 13 bold',  key='-M_PATH-')],
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
        self.window['-WIN_IMG-'].update(data=win_img_data)
        self.window['-LOSE_IMG-'].update(data=lose_img_data)
        self.map_data = mappa_img_data
        self.map_height = img_size_mappa[1]

    def switch_to_pre_game_layout(self):
        self.window['-IN_LAYOUT-'].update(visible=False)
        self.window['-PRE_GAME_LAYOUT-'].update(visible=True)

    def switch_to_game_layout(self, marco_starting_pos):
        self.window['-MARCO_LOCATION_TXT-'].update('Posizione iniziale di Marco:  '+ str(marco_starting_pos))
        self.window['-M_PATH-'].update('Percorso di Marco:  '+ str(marco_starting_pos))
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
        self.window['-D1_START_LOCATION-'].update('Guardiano A:  ' + (str(detective_loc[0]) if detective_loc[0]>9 else (' '+str(detective_loc[0])+' ')))
        self.window['-D2_START_LOCATION-'].update('Guardiano B:  ' + (str(detective_loc[1]) if detective_loc[1]>9 else (' '+str(detective_loc[1])+' ')))
        self.window['-D3_START_LOCATION-'].update('Guardiano C:  ' + (str(detective_loc[2]) if detective_loc[2]>9 else (' '+str(detective_loc[2])+' ')))
        self.window['-D1_LOCATION-'].update('Guardiano A:  ' + (str(detective_loc[0]) if detective_loc[0]>9 else (' '+str(detective_loc[0])+' ')))
        self.window['-D2_LOCATION-'].update('Guardiano B:  ' + (str(detective_loc[1]) if detective_loc[1]>9 else (' '+str(detective_loc[1])+' ')))
        self.window['-D3_LOCATION-'].update('Guardiano C:  ' + (str(detective_loc[2]) if detective_loc[2]>9 else (' '+str(detective_loc[2])+' ')))
        self.window['-D1_PATH-'].update('Percorso del Guardiano A:  ' + (str(detective_loc[0]) if detective_loc[0]>9 else (' '+str(detective_loc[0])+' ')))
        self.window['-D2_PATH-'].update('Percorso del Guardiano B:  ' + (str(detective_loc[1]) if detective_loc[1]>9 else (' '+str(detective_loc[1])+' ')))
        self.window['-D3_PATH-'].update('Percorso del Guardiano C:  ' + (str(detective_loc[2]) if detective_loc[2]>9 else (' '+str(detective_loc[2])+' ')))
        self.window['-PRE_GRAPH-'].draw_image(data=self.map_data, location=(0, self.map_height))
        self.window['-GAME_GRAPH-'].draw_image(data=self.map_data, location=(0, self.map_height))
        for idx,x in enumerate(detective_loc):
            self.window['-PRE_GRAPH-'].draw_circle(self.loc_list[x-1], 15, fill_color=None, line_color='yellow', line_width=4)
            #self.window['-PRE_GRAPH-'].draw_text(self.det_name[idx], self.loc_list[x-1], color='black', angle=0, text_location='center')
            self.window['-GAME_GRAPH-'].draw_circle(self.loc_list[x-1], 15, fill_color=None, line_color='yellow', line_width=4)
            #self.window['-GAME_GRAPH-'].draw_text(self.det_name[idx], self.loc_list[x-1], color='black', angle=0, text_location='center')
        self.detective_starting_loc=detective_loc

    def update_detective(self, detective_loc):
        prev = self.window['-D1_LOCATION-'].get()
        self.window['-D1_LOCATION-'].update(prev + ' --> ' + (str(detective_loc[0]) if detective_loc[0]>9 else (' '+str(detective_loc[0])+' ')))
        prev = self.window['-D1_PATH-'].get()
        self.window['-D1_PATH-'].update(prev + ' --> ' + (str(detective_loc[0]) if detective_loc[0]>9 else (' '+str(detective_loc[0])+' ')))
        prev = self.window['-D2_LOCATION-'].get()
        self.window['-D2_LOCATION-'].update(prev + ' --> ' + (str(detective_loc[1]) if detective_loc[1]>9 else (' '+str(detective_loc[1])+' ')))
        prev = self.window['-D2_PATH-'].get()
        self.window['-D2_PATH-'].update(prev + ' --> ' + (str(detective_loc[1]) if detective_loc[1]>9 else (' '+str(detective_loc[1])+' ')))
        prev = self.window['-D3_LOCATION-'].get()
        self.window['-D3_LOCATION-'].update(prev + ' --> ' + (str(detective_loc[2]) if detective_loc[2]>9 else (' '+str(detective_loc[2])+' ')))
        prev = self.window['-D3_PATH-'].get()
        self.window['-D3_PATH-'].update(prev + ' --> ' + (str(detective_loc[2]) if detective_loc[2]>9 else (' '+str(detective_loc[2])+' ')))
        self.window['-GAME_GRAPH-'].draw_image(data=self.map_data, location=(0, self.map_height))
        for idx,x in enumerate(detective_loc):
            self.window['-GAME_GRAPH-'].draw_circle(self.loc_list[x-1], 15, fill_color=None, line_color='yellow', line_width=4)
            #self.window['-GAME_GRAPH-'].draw_text(self.det_name[idx], self.loc_list[x-1], color='black', angle=0, text_location='center')

    def update_marco_path(self, move):
        if move=='cart':
            transport = 'bici'
        elif move=='tram':
            transport = 'bus'
        else:
            transport = 'nave'
        prev = self.window['-MARCO_LOCATION_TXT-'].get()
        self.window['-MARCO_LOCATION_TXT-'].update(prev + ' -- ' + transport)
        prev = self.window['-M_PATH-'].get()
        self.window['-M_PATH-'].update(prev + ' -- ' + transport)

    def update_counter(self, counter):
        if counter > 10:
          self.window['-COUNTER-'].update('Il gioco è terminato.\nSe non sei stato catturato,\nprendi un mezzo per fuggire!')
        else:
          self.window['-COUNTER-'].update('Turno ' + str(counter))

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
            if not pos.isdigit() or int(pos) not in range(1,22):
                game_gui.window['-MARCO_LOC_INPUT-'].update(text_color='red')
            else:
                mrx_starting_loc = int(pos)
                game_gui.window['-MARCO_LOC_INPUT-'].update(text_color='black')
                game = Game()
                game.initGame(detectives=game_gui.detective_starting_loc, mrX=mrx_starting_loc)
                game_gui.switch_to_game_layout(mrx_starting_loc)

        elif event in ['-BICI-', '-BUS-', '-DELFINO-']:
            if counter > 9:
                game_gui.switch_to_endgame(user_win=True)
            else:
                move = event2move_dict[event]
                game_gui.update_marco_path(move)
                new_detective_loc = game.playTurn(move)
                if new_detective_loc:
                    game_gui.update_detective(new_detective_loc)
                    counter += 1
                    game_gui.update_counter(counter+1)
                else:
                    game_gui.switch_to_endgame(user_win=False)

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