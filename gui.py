#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import tkinter as tk
import warnings
from tkinter import ttk
from tkinter.font import Font

from PIL import Image
from PIL import ImageTk

from logic import Game

warnings.filterwarnings("ignore")

IMG_VARIABILITY_MM: int = 1
IMG_VARIABILITY_CE: int = 2


# noinspection PyAttributeOutsideInit
class GameGUI:
    def __init__(self):
        self.game = Game()
        self.counter = 0
        self.window = tk.Tk()
        self.window.attributes("-fullscreen", True)
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.main_font = Font(family="Tahoma", size=13, weight="bold")
        self.detective_starting_loc = None
        self.w_factor = self.screen_width * 1.1 / 2560
        self.h_factor = self.screen_height * 1.1 / 1440
        self.loc_list = [
            (225, 100),
            (311, 189),
            (412, 205),
            (515, 295),
            (556, 425),
            (505, 133),
            (650, 285),
            (675, 459),
            (563, 530),
            (436, 735),
            (635, 80),
            (730, 336),
            (735, 580),
            (802, 205),
            (826, 300),
            (875, 488),
            (710, 740),
            (852, 60),
            (940, 220),
            (1012, 550),
            (940, 675),
        ]  # 21
        for i in range(len(self.loc_list)):
            self.loc_list[i] = (
                int(self.w_factor * self.loc_list[i][0]),  # type: ignore
                int(0.7 * 0.88 * self.screen_height)
                - int(self.h_factor * self.loc_list[i][1]),
            )

    def create_window(self):
        window_width = int(self.screen_width * 0.88)
        window_height = int(self.screen_height * 0.88)
        self.window.geometry(str(window_width) + "x" + str(window_height))
        self.window.title("La fuga di Marco")
        self.window.resizable(False, False)
        tk.Button(self.window, text=" X ", background="red", command=self.quit).pack(
            side="top", anchor="ne"
        )

        marco_img = [Image.open(f"./img/marco_{i}.jpg") for i in range(IMG_VARIABILITY_MM)]
        alpha_marco = [marco_img[i].size[0] / marco_img[i].size[1] for i in range(IMG_VARIABILITY_MM)]
        img_size_marco = [(int(0.7 * window_height * alpha_marco[i]), int(0.7 * window_height)) for i in range(IMG_VARIABILITY_MM)]
        for i in range(IMG_VARIABILITY_MM):
            marco_img[i] = marco_img[i].resize(img_size_marco[i])

        mappa_img = Image.open("./img/ts_map.jpg")
        alpha_mappa = mappa_img.size[0] / mappa_img.size[1]
        img_size_mappa = (
            int(0.7 * window_height * alpha_mappa),
            int(0.7 * window_height),
        )
        mappa_img = mappa_img.resize(img_size_mappa)

        win_img = [Image.open(f"./img/escaped_{npice}.jpg") for i in range(IMG_VARIABILITY_CE)]
        alpha_win = [win_img[i].size[0] / win_img[i].size[1] for i in range(IMG_VARIABILITY_CE)]
        img_size_win = [(int(0.6 * window_height * alpha_win[i]), int(0.6 * window_height)) for i in range(IMG_VARIABILITY_CE)]
        for i in range(IMG_VARIABILITY_CE):
            win_img[i] = win_img[i].resize(img_size_win[i])

        lose_img = [Image.open(f"./img/captured_{npice}.jpg") for i in range(IMG_VARIABILITY_CE)]
        alpha_lose = [lose_img[i].size[0] / lose_img[i].size[1] for i in range(IMG_VARIABILITY_CE)]
        img_size_lose = [(int(0.6 * window_height * alpha_lose[i]), int(0.6 * window_height)) for i in range(IMG_VARIABILITY_CE)]
        for i in range(IMG_VARIABILITY_CE):
            lose_img[i] = lose_img[i].resize(img_size_lose[i])

        self.marco_img_data = [ImageTk.PhotoImage(marco_img[i]) for i in range(IMG_VARIABILITY_MM)]
        self.win_img_data = [ImageTk.PhotoImage(win_img[i]) for i in range(IMG_VARIABILITY_CE)]
        self.lose_img_data = [ImageTk.PhotoImage(lose_img[i]) for i in range(IMG_VARIABILITY_CE)]
        self.mappa_img_data = ImageTk.PhotoImage(mappa_img)
        self.pos_r = int(img_size_mappa[1] / 37)
        self.pictures = []

        start_txt = "Il pinguino Marco deve scappare dai guardiani dell'acquario di Trieste\nper ottenere pesce dai passanti. Può spostarsi nella città con la bicicletta,\nl'autobus, o il traghetto. Scegli le tue mosse per non farti acchiappare!"
        self.initial_layout = tk.Frame(
            self.window, width=window_width, height=window_height
        )
        tk.Label(
            self.initial_layout, text=start_txt, font=self.main_font, justify="left"
        ).grid(row=0, column=0, columnspan=2, pady=(0, 7))
        self.pictures.append(tk.Label(self.initial_layout, image=self.marco_img_data[random.randint(0, IMG_VARIABILITY_MM)])
        self.pictures[0].grid(  # type: ignore
            row=1, column=0, columnspan=2, pady=7
        ))
        tk.Button(
            self.initial_layout,
            text="Gioca!",
            font=self.main_font,
            width=10,
            height=2,
            background="lightsteelblue",
            command=self.switch_to_pre_game_layout,
        ).grid(row=2, column=0, rowspan=2, pady=7)
        self.mode_select = ttk.Combobox(
            self.initial_layout,
            font=self.main_font,
            width=25,
            textvariable=tk.StringVar(),
            state="readonly",
        )
        self.mode_select["values"] = (
            "Semplice: biglietti illimitati",
            "Medio: biglietti numerati",
            "Difficile: biglietti esatti",
        )
        self.mode_select.current(0)
        self.mode_select.grid(row=2, column=1, padx=16)
        self.detective_ai = ttk.Combobox(
            self.initial_layout,
            font=self.main_font,
            width=25,
            textvariable=tk.StringVar(),
            state="readonly",
        )
        self.detective_ai["values"] = (
            "Contro agenti di RL",
            "Contro agenti probabilistici",
        )
        self.detective_ai.current(0)
        self.detective_ai.grid(row=3, column=1, padx=16)
        self.initial_layout.pack()

        self.pre_game_layout = tk.Frame(
            self.window, width=window_width, height=window_height
        )
        self.canvas = tk.Canvas(
            self.pre_game_layout, width=img_size_mappa[0], height=img_size_mappa[1]
        )
        self.canvas.bind("<Button-1>", self.click_on_pos)
        self.canvas.create_image(0, 0, anchor="nw", image=self.mappa_img_data)
        self.pre_game_pos = (
            self.create_circle(
                self.canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
            self.create_circle(
                self.canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
            self.create_circle(
                self.canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
        )
        self.canvas.grid(row=0, column=0, pady=(0, 7))
        self.pre_game_labels = (
            tk.Label(
                self.pre_game_layout, text="", font=self.main_font, justify="left"
            ),
            tk.Label(
                self.pre_game_layout, text="", font=self.main_font, justify="left"
            ),
            tk.Label(
                self.pre_game_layout, text="", font=self.main_font, justify="left"
            ),
            tk.Label(
                self.pre_game_layout,
                text="Clicca sulla posizione iniziale di Marco",
                font=self.main_font,
                justify="left",
            ),
        )
        for i in range(4):
            self.pre_game_labels[i].grid(row=i + 1, column=0, sticky="w")

        self.game_layout = tk.Frame(
            self.window, width=window_width, height=window_height
        )
        self.map_canvas = tk.Canvas(
            self.game_layout, width=img_size_mappa[0], height=img_size_mappa[1]
        )
        self.map_canvas.create_image(0, 0, anchor="nw", image=self.mappa_img_data)
        self.game_pos = (
            self.create_circle(
                self.map_canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
            self.create_circle(
                self.map_canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
            self.create_circle(
                self.map_canvas, self.pos_r + 1, self.pos_r + 1, self.pos_r, "yellow"
            ),
        )

        self.map_canvas.grid(row=0, column=0, columnspan=5, pady=(0, 7))
        self.game_labels = (
            tk.Label(
                self.game_layout, text="", font=self.main_font, justify="left", width=24
            ),
            tk.Label(self.game_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.game_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.game_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.game_layout, text="", font=self.main_font, justify="left"),
        )
        for i in range(1, 5):
            self.game_labels[i].grid(row=i, column=0, columnspan=5, sticky="w")
        self.game_buttons = (
            tk.Button(
                self.game_layout,
                text="",
                font=self.main_font,
                width=12,
                height=3,
                background="deep sky blue",
                command=lambda: self.click_move("bike"),
            ),
            tk.Button(
                self.game_layout,
                text="",
                font=self.main_font,
                width=12,
                height=3,
                background="red",
                command=lambda: self.click_move("bus"),
            ),
            tk.Button(
                self.game_layout,
                text="",
                font=self.main_font,
                width=12,
                height=3,
                background="green3",
                command=lambda: self.click_move("boat"),
            ),
            tk.Button(
                self.game_layout,
                text="CATTURATO!",
                font=self.main_font,
                width=12,
                height=3,
                background="lightgray",
                command=lambda: self.switch_to_endgame(False),
            ),
        )
        for i in range(4):
            self.game_buttons[i].grid(row=5, column=i, sticky="w", pady=7, padx=(0, 12))
        self.game_labels[0].grid(row=5, column=4, sticky="w")

        win_txt = "Congratulazioni, sei riuscito a fuggire! Ti sei meritato un sacco di pesce!"
        self.win_layout = tk.Frame(
            self.window, width=window_width, height=window_height
        )
        tk.Label(
            self.win_layout, text=win_txt, font=self.main_font, justify="left"
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.pictures.append(tk.Label(self.win_layout, image=self.win_img_data[random.randint(0, IMG_VARIABILITY_CE)]))
        self.pictures[1].grid(  # type: ignore
            row=1, column=0, sticky="w", pady=7
        )
        tk.Button(
            self.win_layout,
            text="Ricomincia",
            font=self.main_font,
            width=16,
            height=2,
            background="lightsteelblue",
            command=lambda: self.restart_layout(True),
        ).grid(row=2, column=0, pady=10)

        lose_txt = "Oh no, sei stato catturato! Si ritorna all'acquario!"
        self.lose_layout = tk.Frame(
            self.window, width=window_width, height=window_height
        )
        tk.Label(
            self.lose_layout, text=lose_txt, font=self.main_font, justify="left"
        ).grid(row=0, column=0, sticky="w", pady=(0, 0))
        self.path_labels = (
            tk.Label(self.lose_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.lose_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.lose_layout, text="", font=self.main_font, justify="left"),
            tk.Label(self.lose_layout, text="", font=self.main_font, justify="left"),
        )
        for i in range(4):
            self.path_labels[i].grid(row=i + 1, column=0, sticky="w")
        self.pictures.append(tk.Label(self.lose_layout, image=self.lose_img_data[random.randint(0, IMG_VARIABILITY_CE)]))
        self.pictures[2].grid(  # type: ignore
            row=5, column=0, sticky="w", pady=7
        )
        tk.Button(
            self.lose_layout,
            text="Ricomincia",
            font=self.main_font,
            width=16,
            height=2,
            background="lightsteelblue",
            command=lambda: self.restart_layout(False),
        ).grid(row=6, column=0, pady=7)

    @staticmethod
    def create_circle(canv, x, y, r, color):
        return canv.create_oval(
            x - r, y - r, x + r, y + r, fill=None, outline=color, width=4
        )

    def nearest_loc(self, x, y):
        dist = 1200  # initially: max acceptable distance
        loc = None
        for i in range(len(self.loc_list)):
            new_dist = ((self.loc_list[i][0] - x) / self.w_factor) ** 2 + (
                (self.loc_list[i][1] - y) / self.h_factor
            ) ** 2
            if new_dist < dist:
                dist = new_dist
                loc = i + 1
        return loc

    def click_on_pos(self, event):
        mrx_starting_loc = self.nearest_loc(event.x, event.y)
        if mrx_starting_loc and (mrx_starting_loc not in self.detective_starting_loc):
            self.switch_to_game_layout(mrx_starting_loc)

    def click_move(self, move):
        if self.counter == 10:
            self.switch_to_endgame(True)
        else:
            if self.mode_select.current() > 0:
                if move == "bike":
                    self.tickets[0] -= 1
                elif move == "bus":
                    self.tickets[1] -= 1
                elif move == "boat":
                    self.tickets[2] -= 1
            self.update_marco_path(move)
            new_detective_loc, found = self.game.playTurn(move)
            if new_detective_loc:
                self.update_detective(new_detective_loc)
                self.counter += 1
                self.update_counters()
            else:
                self.switch_to_endgame(False)
            if found:
                self.game_buttons[0].configure(text="Bicicletta", state="disabled")
                self.game_buttons[1].configure(text="Autobus", state="disabled")
                self.game_buttons[2].configure(text="Traghetto", state="disabled")
                for i in range(3):
                    self.game_buttons[i].update()

    def switch_to_pre_game_layout(self):
        detective_loc = random.sample(range(1, 22), 3)
        self.set_detective_starting_loc(detective_loc)
        # self.update_infoSet([])
        self.initial_layout.pack_forget()
        self.pre_game_layout.pack()

    def switch_to_game_layout(self, marco_starting_pos):
        self.counter = 0
        self.game.initGame(
            self.detective_starting_loc,
            marco_starting_pos,
            not bool(self.detective_ai.current()),
        )

        self.game_labels[0].configure(text="Turno 1")
        self.game_labels[4].configure(
            text="Posizione iniziale di Marco:  " + str(marco_starting_pos)
        )
        self.path_labels[3].configure(
            text="Percorso di Marco:  " + str(marco_starting_pos)
        )
        if self.mode_select.current() == 0:
            self.tickets = None
            self.game_buttons[0].configure(text="Bicicletta", state="normal")
            self.game_buttons[1].configure(text="Autobus", state="normal")
            self.game_buttons[2].configure(text="Traghetto", state="normal")
        else:
            if self.mode_select.current() == 1:
                self.tickets = [8, 5, 3]
            else:
                self.tickets = [6, 3, 1]
            self.game_buttons[0].configure(
                text=f"Bicicletta\n({self.tickets[0]})", state="normal"
            )
            self.game_buttons[1].configure(
                text=f"Autobus\n({self.tickets[1]})", state="normal"
            )
            self.game_buttons[2].configure(
                text=f"Traghetto\n({self.tickets[2]})", state="normal"
            )
        for i in range(3):
            self.game_buttons[i].update()
        self.game_labels[0].update()
        self.game_labels[4].update()
        self.path_labels[3].update()
        self.pre_game_layout.pack_forget()
        self.game_layout.pack()

    def switch_to_endgame(self, win):
        self.game_layout.pack_forget()
        if win:
            self.win_layout.pack()
        else:
            self.lose_layout.pack()

    def set_detective_starting_loc(self, detective_loc):
        self.detective_starting_loc = detective_loc
        self.pre_game_labels[0].configure(
            text="Guardiano A:  "
            + (
                str(detective_loc[0])
                if detective_loc[0] > 9
                else (" " + str(detective_loc[0]) + " ")
            )
        )
        self.pre_game_labels[1].configure(
            text="Guardiano B:  "
            + (
                str(detective_loc[1])
                if detective_loc[1] > 9
                else (" " + str(detective_loc[1]) + " ")
            )
        )
        self.pre_game_labels[2].configure(
            text="Guardiano C:  "
            + (
                str(detective_loc[2])
                if detective_loc[2] > 9
                else (" " + str(detective_loc[2]) + " ")
            )
        )
        self.game_labels[1].configure(
            text="Guardiano A:  "
            + (
                str(detective_loc[0])
                if detective_loc[0] > 9
                else (" " + str(detective_loc[0]) + " ")
            )
        )
        self.game_labels[2].configure(
            text="Guardiano B:  "
            + (
                str(detective_loc[1])
                if detective_loc[1] > 9
                else (" " + str(detective_loc[1]) + " ")
            )
        )
        self.game_labels[3].configure(
            text="Guardiano C:  "
            + (
                str(detective_loc[2])
                if detective_loc[2] > 9
                else (" " + str(detective_loc[2]) + " ")
            )
        )
        self.path_labels[0].configure(
            text="Percorso del Guardiano A:  "
            + (
                str(detective_loc[0])
                if detective_loc[0] > 9
                else (" " + str(detective_loc[0]) + " ")
            )
        )
        self.path_labels[1].configure(
            text="Percorso del Guardiano B:  "
            + (
                str(detective_loc[1])
                if detective_loc[1] > 9
                else (" " + str(detective_loc[1]) + " ")
            )
        )
        self.path_labels[2].configure(
            text="Percorso del Guardiano C:  "
            + (
                str(detective_loc[2])
                if detective_loc[2] > 9
                else (" " + str(detective_loc[2]) + " ")
            )
        )
        for i in range(3):
            self.pre_game_labels[i].update()
            self.game_labels[i + 1].update()
            self.path_labels[i].update()
        for idx, x in enumerate(detective_loc):
            self.canvas.moveto(
                self.pre_game_pos[idx],
                self.loc_list[x - 1][0] - self.pos_r - 2,
                self.loc_list[x - 1][1] - self.pos_r - 2,
            )
            self.map_canvas.moveto(
                self.game_pos[idx],
                self.loc_list[x - 1][0] - self.pos_r - 2,
                self.loc_list[x - 1][1] - self.pos_r - 2,
            )

    def update_detective(self, detective_loc):
        for i in range(3):
            new = " → " + (
                str(detective_loc[i])
                if detective_loc[i] > 9
                else (" " + str(detective_loc[i]) + " ")
            )
            prev = self.game_labels[i + 1]["text"]
            self.game_labels[i + 1].configure(text=prev + new)
            prev = self.path_labels[i]["text"]
            self.path_labels[i].configure(text=prev + new)
        for i in range(3):
            self.game_labels[i + 1].update()
            self.path_labels[i].update()
        for idx, x in enumerate(detective_loc):
            self.map_canvas.moveto(
                self.game_pos[idx],
                self.loc_list[x - 1][0] - self.pos_r - 2,
                self.loc_list[x - 1][1] - self.pos_r - 2,
            )

    def update_marco_path(self, move):
        if move == "bike":
            transport = "bici"
        elif move == "bus":
            transport = "bus"
        else:
            transport = "nave"
        prev = self.game_labels[4]["text"]
        self.game_labels[4].configure(text=prev + " — " + transport)
        prev = self.path_labels[3]["text"]
        self.path_labels[3].configure(text=prev + " — " + transport)
        self.game_labels[4].update()
        self.path_labels[3].update()

    def update_counters(self):
        if self.counter == 10:
            self.game_labels[0].configure(
                text="Il gioco è terminato.\nSe non sei stato catturato,\nprendi un mezzo per fuggire!"
            )
            if self.mode_select.current() > 0:
                self.game_buttons[0].configure(text="Bicicletta", state="normal")
                self.game_buttons[1].configure(text="Autobus", state="normal")
                self.game_buttons[2].configure(text="Traghetto", state="normal")
                for i in range(3):
                    self.game_buttons[i].update()
        else:
            self.game_labels[0].configure(text="Turno " + str(self.counter + 1))
            if self.mode_select.current() > 0:
                self.game_buttons[0].configure(
                    text=f"Bicicletta\n({self.tickets[0]})",
                    state="normal" if self.tickets[0] else "disabled",
                )
                self.game_buttons[1].configure(
                    text=f"Autobus\n({self.tickets[1]})",
                    state="normal" if self.tickets[1] else "disabled",
                )
                self.game_buttons[2].configure(
                    text=f"Traghetto\n({self.tickets[2]})",
                    state="normal" if self.tickets[2] else "disabled",
                )
                for i in range(3):
                    self.game_buttons[i].update()
        self.game_labels[0].update()

    def restart_layout(self, win):
        if win:
            self.win_layout.pack_forget()
        else:
            self.lose_layout.pack_forget()
        self.pictures[0].configure(image=self.marco_img_data[random.randint(0, IMG_VARIABILITY_MM)])
        self.pictures[1].configure(image=self.win_img_data[random.randint(0, IMG_VARIABILITY_CE)])
        self.pictures[2].configure(image=self.lose_img_data[random.randint(0, IMG_VARIABILITY_CE)])
        for i in range(3):
            self.pictures[i].update()
        self.initial_layout.pack()

    def mainloop(self):
        self.window.mainloop()

    def quit(self):
        self.window.destroy()


if __name__ == "__main__":
    game_gui = GameGUI()
    game_gui.create_window()
    game_gui.mainloop()
