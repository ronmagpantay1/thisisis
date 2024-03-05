"""
THis program recognizes pitches in real time.
"""
import math
import queue
import threading
import time
import tkinter as tk
import wave
import librosa.feature
import numpy as np
import pyaudio
import ctypes





hz_array = np.array([32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49.0, 51.91, 55.0, 58.27, 61.74,
                     65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98.0, 103.83, 110.0, 116.54, 123.47,
                     130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.0, 196.0, 207.65, 220.0, 233.08, 246.94,
                     261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.0, 415.3, 440.0, 466.16, 493.88,
                     523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.0, 932.33, 987.77,
                     1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.0, 1864.66,
                     1975.53, 2093.0])

hz_notename = {32.7: 'C1', 34.65: '#C1', 36.71: 'D1', 38.89: '#D1', 41.2: 'E1', 43.65: 'F1',
               46.25: '#F1', 49.0: 'G1', 51.91: '#G1', 55.0: 'A1', 58.27: '#A1', 61.74: 'B1',
               65.41: 'C2', 69.3: '#C2', 73.42: 'D2', 77.78: '#D2', 82.41: 'E2', 87.31: 'F2',
               92.5: '#F2', 98.0: 'G2', 103.83: '#G2', 110.0: 'A2', 116.54: '#A2', 123.47: 'B2',
               130.81: 'C3', 138.59: '#C3', 146.83: 'D3', 155.56: '#D3', 164.81: 'E3', 174.61: 'F3',
               185.0: '#F3', 196.0: 'G3', 207.65: '#G3', 220.0: 'A3', 233.08: '#A3', 246.94: 'B3',
               261.63: 'C4', 277.18: '#C4', 293.66: 'D4', 311.13: '#D4', 329.63: 'E4', 349.23: 'F4',
               369.99: '#F4', 392.0: 'G4', 415.3: '#G4', 440.0: 'A4', 466.16: '#A4', 493.88: 'B4',
               523.25: 'C5', 554.37: '#C5', 587.33: 'D5', 622.25: '#D5', 659.25: 'E5', 698.46: 'F5',
               739.99: '#F5', 783.99: 'G5', 830.61: '#G5', 880.0: 'A5', 932.33: '#A5', 987.77: 'B5',
               1046.5: 'C6', 1108.73: '#C6', 1174.66: 'D6', 1244.51: '#D6', 1318.51: 'E6', 1396.91: 'F6',
               1479.98: '#F6', 1567.98: 'G6', 1661.22: '#G6', 1760.0: 'A6', 1864.66: '#A6', 1975.53: 'B6', 2093.0: 'C7'}

 #...[C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
easy_chord_template = {
    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'Cm': [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'C#': [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'D#': [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'F#': [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'G#': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#': [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    'Bbm': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'NC': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}


def find_nearest(array, value):
    """
    This function returns the number in a linear array that is closest to the given number.
    This function returns the number closest to the parameter value in the numpy one-dimensional array.
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return array[idx - 1]
    else:
        return array[idx]


# Display note names
def title(title_text_var):
    def _title(belonging, master, position, font_size, expand, wid, hei, padx=0, pady=0):
        _title_v = tk.Label(
            master,
            textvariable=title_text_var,  # tagtext
            bg=BG_COLOUR,  # background color
            font=('Arial', font_size),  # fontsize
            width=wid, height=hei)  # label,width,length
        _title_v.pack(side=position, padx=padx, pady=pady, expand=expand)  # fix window position
        if belonging is not None:
            belonging.append(_title_v)
        title_text_var.set('')

    return _title


def cossim(u, v):
    """
    :param u: non-negative vector u
    :param v: non-negative vector v
    :return: the cosine similarity between u and v
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def chordgram(filename, sr, hop_length):
    """
    read audio files and use CQT algorithm to convert them into chromagram
    :param filename: file path
    :param sr: sampling rate
    :param hop_length: number of samples between consecutive chroma frames (frame size)
    :return: chromagram
    """
    """
    Start to calculate chromagram
    """
    y, _sr = librosa.load(filename, sr=sr)
    # harmonic content extraction
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Constant Q Transform
    st = time.time()
    _chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=_sr, hop_length=hop_length)
    print(time.time() - st)

    """
    Start to calculate chordgram
    """
    _frames = _chromagram.shape[1]

    # initialize
    chords = list(chord_template.keys())
    chroma_vectors = np.transpose(_chromagram)  # matrix transpose
    # _chordgram
    _chordgram = []

    for n in np.arange(_frames):
        cr = chroma_vectors[n]
        sims = []

        for chord in chords:
            chord_tpl = chord_template[chord]
            # calculate cos sim, add weight
            if chord == "NC":
                sim = cossim(cr, chord_tpl) * 0.8
            else:
                sim = cossim(cr, chord_tpl)
            sims += [sim]
        _chordgram += [sims]
    _chordgram = np.transpose(_chordgram)
    return np.array(_chordgram)



def chord_sequence(_chordgram):
    """
    :param _chordgram: H
    :return: a sequence of chords
    """
    chords = list(chord_template.keys())
    _frames = _chordgram.shape[1]
    _chordgram = np.transpose(_chordgram)
    # print(_chordgram.shape)  # -> (1, 13)
    chordset = []

    for n in np.arange(_frames):
        index = np.argmax(_chordgram[n])
        if _chordgram[n][index] == 0.0:
            chord = "NC"
        else:
            chord = chords[index]

        chordset += [chord]
    return chordset


def note_recognize(_note_text_var):
    # Calculate note frequency using fast Fourier transform
    y, sr = librosa.load('tmp.wav')  # save as wave file first.
    _f0 = librosa.yin(y, fmin=60, fmax=400)  # transform
    _f0[np.isnan(_f0)] = 0  # replace the null value nan with 0
    _f0 = [find_nearest(hz_array, i) for i in list(_f0)]  # Convert the frequency to the frequency of the nearest standard tone

    # TODO:下Filter out the notes below. (Note that only single notes can be recognized, chords cannot be recognized temporarily)
    # Add the standard frequency value to the list latest notes list
    for i in _f0:
        latest_notes_list.append(float(i))
    # Take 6 frequency data and conduct an overall analysis of the sounds within the group
    if len(latest_notes_list) < 5:
        pass
    else:
        max_note = max(latest_notes_list, key=latest_notes_list.count)  # Take the frequency with the largest number
        if latest_notes_list.count(max_note) > 1:  # Only the frequency with the largest number > 2 will be output
            """
            The frequency obtained by the program with the largest number is not necessarily the fundamental frequency. It may be an overtone or a fifth. If these notes are also in the latest_note_list list, this note will be displayed.
            First, take the frequencies of these sounds out of the hz array:
            (Get the index of max_note in hz array, and then use the value obtained by subtracting the interval from the index as the index of the new sound to find these sounds)
            """
            f0_possible2 = hz_array[list(hz_array).index(max_note) - 12]  # First overtone (octave)
            global LOADING, load_str_var
            if LOADING:
                load_str_var.set("Complete!")
                time.sleep(0.5)
                text_load.destroy()
                LOADING = 0
            if f0_possible2 in latest_notes_list:  # The first overtone is the most likely
                _note_text_var.set(str(hz_notename[f0_possible2]))
            else:  # If the above tone is not in the list, the tone with the largest number is displayed.
                _note_text_var.set(str(hz_notename[max_note]))
        else:  # If there are not more than two notes, the notes displayed in the window remain unchanged until the next note is judged.
            _note_text_var.set('NN')  # ps: If you write here, there will be blank spaces that flicker from time to time (because the window text is redrawn), which looks too ugly/doge
            pass
        latest_notes_list.clear()


# Get the recorded data and put it into q
def audio_callback(in_data, *args):
    str(args)
    q.put(in_data)
    ad_rdy_ev.set()
    return None, pyaudio.paContinue


def read_audio_thread(_q, _stream, _frames, _ad_rdy_ev):
    global latest_notes_list, both_title_v, text_load
    while _stream.is_active():
        # _ad_rdy_ev.wait(timeout=1000)
        # if _q.empty():
        #     _ad_rdy_ev.clear()
        #     print("break")
        #     break
        _data = _q.get()
        # Get the audio stream data from _q (_data looks like the data of the wave file)
        while not _q.empty():
            _q.get()

        # Save data to wav file
        wave_data = b"".join([_data])
        with wave.open("tmp.wav", "wb") as wf1:
            wf1.setnchannels(CHANNELS)
            wf1.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf1.setframerate(RATE)
            wf1.writeframes(wave_data)
        global mode
        if mode == "chord":
            try:
                both_title_v.forget()
            except NameError:
                pass
            chordset = chord_sequence(chordgram("tmp.wav", sr=44100, hop_length=4096))
            print(chordset)
            global LOADING, load_str_var
            if LOADING:
                load_str_var.set("Complete!")
                time.sleep(0.5)
                text_load.destroy()
                LOADING = 0
            if chordset[0] == chordset[1] == "NC":
                note_text_var.set("NC")
            else:
                for i in chordset:
                    if i != "NC":
                        note_text_var.set(i)
                # note_text_var.set(chordset[1]) if chordset[1] != "NC" else note_text_var.set(chordset[0])
        elif mode == "note":
            try:
                both_title_v.forget()
            except NameError:
                pass
            note_recognize(note_text_var)
        elif mode == "both":
            both_title_v.pack(side="right", expand=True, padx=0, pady=10)
            note_recognize(note_text_var2)
            chordset = chord_sequence(chordgram("tmp.wav", sr=44100, hop_length=4096))
            print(chordset)
            if chordset[0] == chordset[1] == "NC":
                note_text_var.set("NC")
            else:
                for i in chordset:
                    if i != "NC":
                        note_text_var.set(i)
        # if Recording:
        #     _frames.append(_data)
        # print(_frames)
        _ad_rdy_ev.clear()


def draw_main_window():
    global both_title_v, text_load

    title_v = tk.Label(
        window_,
        text='ChorDefine',
        bg='black', fg="white",
        font=('Arial', 36),
        width=30, height=2,
        relief="raised", bd="5", )
    title_v.pack()

    # frame0
    f0 = tk.Frame(master=window_, bg='white')
    f0.pack(side="top", fill="x")
    window_widgets.append(f0)
    # frame1
    f1 = tk.Frame(master=f0, padx=10, pady=5, bg='black')
    f1.pack(side="top")
    window_widgets.append(f1)

    # dividing line
    ff = tk.Frame(master=window_, bg='black')
    ff.pack(side="top", fill="x", pady=2)
    window_widgets.append(ff)

    # Mode of recognition
    text_var_mode = tk.StringVar()
    title(text_var_mode)(
        window_widgets, f1, position="left", expand=False, font_size=25, wid=10, hei=1, padx=5)
    text_var_mode.set("Mode:")

    def radiobutton_go():
        current_value = var.get()

        if current_value == 'chord':
            radio_button_chord.config(bg='white', fg='black')
            radio_button_note.config(bg='black', fg='white')
            radio_button_both.config(bg='black', fg='white')
        elif current_value == 'note':
            radio_button_chord.config(bg='black', fg='white')
            radio_button_note.config(bg='white', fg='black')
            radio_button_both.config(bg='black', fg='white')
        elif current_value == 'both':
            radio_button_chord.config(bg='black', fg='white')
            radio_button_note.config(bg='black', fg='white')
            radio_button_both.config(bg='white', fg='black')

        global mode
        mode = var.get()
    var = tk.StringVar()

    radio_button_chord = tk.Radiobutton(
        f1, text="Chord", bg='black', variable=var, indicatoron=False,
        font=('Arial', 20), value='chord', width=8, padx=3, command=radiobutton_go, fg='white')
    radio_button_note = tk.Radiobutton(
        f1, text="Note", bg='black', variable=var, indicatoron=False,
        font=('Arial', 20), value='note', width=8, padx=3, command=radiobutton_go, fg='white')
    radio_button_both = tk.Radiobutton(
        f1, text="Both", bg='black', variable=var, indicatoron=False,
        font=('Arial', 20), value='both', width=8, padx=3, command=radiobutton_go, fg='white')

    radio_button_chord.pack(side="left", expand=False, padx=5)
    radio_button_note.pack(side="left", expand=False, padx=5)
    radio_button_both.pack(side="left", expand=False, padx=5)

    # loading
    text_load = tk.Label(
        window_,
        textvariable=load_str_var,
        bg='black',
        font=('Arial', 70),
        width=10, height=4, fg='white')
    text_load.pack(side="top", padx=5, pady=10, expand=True)
    load_str_var.set("...Loading...")

    # Output character variables of note and chord types
    # frame2
    f2 = tk.Frame(master=window_, padx=10, pady=5, bg='white',)
    f2.pack(side="top", fill="both")
    window_widgets.append(f2)

    title(note_text_var)(window_widgets, f2, position="left", expand=True, font_size=160, wid=4, hei=2, pady=10)
    both_title_v = tk.Label(
        master=f2,
        textvariable=note_text_var2,
        bg="black",
        font=('Arial', 160), fg='white',
        width=4, height=2)
    window_.mainloop()



if __name__ == "__main__":
    Recording = False  # to record or not
    mode = "chord"
    # basic audio parameters
    CHUNK = 2048
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    WAVE_OUTPUT_FILENAME = "output.wav"
    chord_template = easy_chord_template
    latest_notes_list = []
    frames = []

    p = pyaudio.PyAudio()  # pyaudio obj
    q = queue.Queue()  # A "queue" that accepts audio stream data

    # audio data stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=False,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    wf = None
    if Recording:
        # Read the file and set basic parameters
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

    # Turn on the audio stream and start recording
    stream.start_stream()

    ad_rdy_ev = threading.Event()
    thread_ = threading.Thread(target=read_audio_thread, args=(q, stream, frames, ad_rdy_ev))  # 新线程，用于分析得到的音频数据
    thread_.daemon = True
    thread_.start()

    # Tkinter
    BG_COLOUR = 'Beige'
    window_ = tk.Tk()
    window_.title('')
    # window_.attributes('-fullscreen', True)
    window_.geometry('1366x768')
    window_.minsize(1100, 550)
    window_.maxsize(1920, 1080)
    window_.configure(cursor="arrow", height=80, bg=BG_COLOUR)
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Tell the operating system to use the program's own dpi adaptation
    ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)  # Get the zoom factor of the screen
    window_.tk.call('tk', 'scaling', ScaleFactor / 75)  # Set program scaling

    window_widgets = []  # Initialize the control list
    text_load = tk.Label()
    LOADING = True
    load_str_var = tk.StringVar()
    note_text_var = tk.StringVar()  # Initialize global controls
    note_text_var2 = tk.StringVar()
    both_title_v = tk.Label()
    draw_main_window()

    # After exiting the window, stop and close the audio stream and delete the pyaudio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("")

    # if Recording:
    #     wf.writeframes(b''.join(frames))
    #     wf.close()
