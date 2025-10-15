# %% imports
from imports3 import *
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
# %% load data
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")
# %% data overview
print(X.shape, Y.shape)
print(X.columns)
print(X.loc[1, 'Skeleton_Sequence'].shape)
print(Y)

Patient_Exercise_counts = X[['Patient_Id','Exercise_Id']].groupby('Patient_Id')['Exercise_Id'].value_counts().unstack()
Patient_Exercise_counts['Total'] = Patient_Exercise_counts.sum(axis=1)
print(Patient_Exercise_counts)
vals, counts = np.unique(Y, return_counts=True)
side_df = pd.DataFrame(columns=['left stroke','right stroke'])
side_df.loc[0] = counts
print(side_df)
# %%
for i in range(14):
    print(f'Patient {i+1}: {stroke_dict[Y[i]]}')


'''
X shape (444, 3), Y shape (14,)
X columns: Patient_Id (int), Exercise_Id (object ex: E1), Skeleton_Sequence (array, ex: (172,66))
66 features for x and y coords? different number of frames (rows)
Y is just 0's and 1's

Patients did a different number of exercises each and total.
Unbalanced targets

E1 - brushing hair
E2 - brushing teeth
E3 - washing face
E4 - put on socks
E5 - hip flexion
'''
# %% keypoints
all_keypoints = {'r':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'l':[0,1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]}
hand_keypoints = {'l':[15,17,19,21], 'r':[16,18,20,22]}
leg_keypoints = {'l':[23,25,27,29,31], 'r':[24,26,28,30,32]}
torso_keypoints = {'l':[11,23], 'r':[12,24]}
face_keypoints = {'l':[7,3,2,1,0], 'r':[4,5,6,8]}
# %% functions
def make_cols(indexes, components=['x','y']):
    col_names = [c+str(i) for i in indexes for c in components]
    return col_names
def skeleton_sequence_to_df(X_ss):
    col_names = make_cols(range(33))
    df = pd.DataFrame(X_ss, columns=col_names)
    return df

def ss_animation(ax, X_ss, interval=50):
    plot_keypoints = [
        [28,30,32,28,26,24,12,14,16,18,20,16,22,16,14,12,11,13,15,17,19,15,21,15,13,11,23,24,23,25,27,29,31,27],
        [9,10],
        [8,6,5,4,0,1,2,3,7],
    ]

    df = skeleton_sequence_to_df(X_ss)  # expects columns x0.., y0.. per frame row
    n_frames = len(df)

    lines = [ax.plot([], [], lw=2)[0] for _ in plot_keypoints]
    ax.grid(True)

    # overlay text (axes coords)
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                   fontsize=10, va="top", ha="left", color="black")

    # limits (pad 10%)
    dmin = 1.1 * min(df.filter(like='x').min().min(), df.filter(like='y').min().min())
    dmax = 1.1 * max(df.filter(like='x').max().max(), df.filter(like='y').max().max())
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(-dmax, -dmin)  # flip if your coords are screen-like
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title("Skeleton Animation")
    ax.set_aspect("equal", adjustable="box")

    # precompute column names
    xcols = [[f'x{kp}' for kp in ks] for ks in plot_keypoints]
    ycols = [[f'y{kp}' for kp in ks] for ks in plot_keypoints]

    # drive frames manually
    current = [0]  # mutable so inner funcs can modify

    def _draw_frame(i):
        row = df.iloc[i]
        for line, xc, yc in zip(lines, xcols, ycols):
            x = row[xc].values
            y = -row[yc].values
            line.set_data(x, y)
        text.set_text(f"frame: {i}")
        return (*lines, text)

    def init():
        for line in lines:
            line.set_data([], [])
        text.set_text("frame: 0")
        return (*lines, text)

    def update(_tick):
        # if finished, hold last frame and stop timer
        if current[0] >= n_frames:
            # already finished: ensure we display the last frame and stop
            _draw_frame(n_frames - 1)
            ani.event_source.stop()
            return (*lines, text)

        # draw current frame and advance
        artists = _draw_frame(current[0])
        current[0] += 1
        return artists

    # Use frames=None so matplotlib uses an endless counter; we stop it ourselves.
    ani = FuncAnimation(ax.figure, update, init_func=init,
                        frames=None, interval=interval, blit=True, repeat=False)

    # ---- Restart button (per subplot) ----
    # small button just below the axes (normalized figure coords)
    bbox = ax.get_position()
    btn_ax = ax.figure.add_axes([bbox.x0 + 0.02, bbox.y0, 0.02, 0.03])
    restart_btn = Button(btn_ax, '⟳', color='0.9', hovercolor='0.95')
    restart_btn.label.set_fontsize(8)

    def restart(event=None):
        current[0] = 0            # reset our frame counter
        ani.event_source.stop()   # ensure timer is stopped
        init()                    # (optional) immediately clear/redraw first frame
        ani.event_source.start()  # start ticking again

    restart_btn.on_clicked(restart)

    return ani, (restart_btn, restart)

stroke_dict = {0:'left stroke', 1:'right stroke'}
# %% visualize skeleton sequence animation
patients = [3, 4 ,5]
targets = ['E1']
patient_ids = [id for _ in range(len(targets)) for id in patients ]
targets_class = [t for t in targets for _ in range(len(patients))]

anis = []
buttons = []
restart_fs = []
def plot_f(ax, i):
    patient_id = patient_ids[i]
    target_id  = targets_class[i]
    X_plot = X.query(f'Patient_Id=={patient_id} & Exercise_Id=="{target_id}"')
    X_plot = X_plot.iloc[0]
    row = X_plot.name
    

    ani, (btn, restart_f) = ss_animation(ax, X_plot['Skeleton_Sequence'], interval=50)
    anis.append(ani)
    buttons.append(btn)
    restart_fs.append(restart_f)
    ax.set_title(f'Patient {patient_id}, Exercise {target_id}, {stroke_dict[Y[patient_id-1]]}')
    lines = ax.get_lines()
    for i, line in enumerate(lines):
        color = f"C{patient_id}"  # use Matplotlib’s default color cycle
        line.set_color(color)

fig ,axes = min_multiple_plot(len(patient_ids), plot_f, n_cols=len(patients))
#restart all animations button
if 1:
    btn_ax = fig.add_axes([0.02, 0.02, 0.02, 0.03])
    restart_btn = Button(btn_ax, '⟳ all', color='0.9', hovercolor='0.95')
    restart_btn.label.set_fontsize(8)

    def on_master(event):
        for rf in restart_fs:
            rf(event)
    restart_btn.on_clicked(on_master)
#plt.tight_layout()
plt.show()


'''
Ha videos em que os pacientes começam em diferentes fases do exercicio e repetem um diferente numero de vezes
Tentar separar em mais amostras as repeticoes?

Arranjar diferentes metodos para cada tipo de exercicio
Exercicios em que ha simetria - contar o numero de frames que demora em cada lado
O numero de frames que o paciente demora a completar o exercicio também é uma medida do quao afetado o paciente esta - mas os pacientes começam
em diferentes fases do exercicio e fazem um diferente numero de vezes

E2 - Paciente 2 faz movimentos muito mais largos por falta de coordenacao!

paciente 1 E1 - what is he doing?!?!
'''


# %% Time flatten visual of keypoints

row_index = 0
patients = [3,4,5]
targets = ['E1']
patient_ids = [id for _ in range(len(targets)) for id in patients ]
targets_class = [t for t in targets for _ in range(len(patients))]

def plot_f(ax, i):
    patient_id = patient_ids[i]
    target_id  = targets_class[i]
    X_plot = X.query(f'Patient_Id=={patient_id} & Exercise_Id=="{target_id}"')
    X_plot = X_plot.iloc[row_index]
    row = X_plot.name
    
    df = skeleton_sequence_to_df(X_plot['Skeleton_Sequence'])
    keypoint_index_dict = {19:'l_index',20:'r_index'}
    keypoint_index_color_dict = {19:plt.cm.grey,20:plt.cm.viridis}
    for index in [19,20]:
        cols = make_cols([index])
        x_cols = [c for c in cols if 'x' in c]
        y_cols = [c for c in cols if 'y' in c]
        # Assume df[x_cols] and df[y_cols] are 1D (or take first col if multiple)
        x = df[x_cols].values.flatten()
        y = -df[y_cols].values.flatten()

        # Create a hue that increases with point index
        t = np.arange(len(x))  # goes from 0 to N-1
        colors = keypoint_index_color_dict[index](t / t.max())  # normalize and map to a colormap
        # Scatter plot with color gradient
        ax.scatter(x, y, c=colors, s=30, label=keypoint_index_dict[index])
        ax.plot(x, y, color='gray', alpha=0.5)  # optional: connect points with a line
   # ax.text(min(x)*1.0, min(y)*1.0, stroke_dict[Y[patient_id-1]])
    ax.grid(True)
    ax.legend()
    ax.set_title(f'Patient {patient_id}, {target_id}, {stroke_dict[Y[patient_id-1]]}')
    ax.set_aspect('equal')

fig ,axes = min_multiple_plot(len(patient_ids), plot_f, n_cols=len(patients))
plt.show()

'''
Patient 1, E4, left stroke - Its very hard to distinguish stroke side - Maybe its important to create a measure of impairment


'''

# %%

patient_ids = [1,2, 5,6]
target_id  = 'E4'
y_max = 0
dfs = []
fig, axes = plt.subplots(len(patient_ids), 2)
for patient_id in patient_ids:
    X_plot = X.query(f'Patient_Id=={patient_id} & Exercise_Id=="{target_id}"')
    X_plot = X_plot.iloc[0]
    df_ = skeleton_sequence_to_df(X_plot['Skeleton_Sequence']).copy()
    df_total_distance = pd.DataFrame()
    
    for key, indexes in all_keypoints.items():
        cols = make_cols(indexes)

        for col in cols:
            df_[col+'diff'] = df_[col].diff().fillna(0)
        
        for index in indexes:
            df_[str(index)+'dist'] = np.sqrt(df_[f'x{index}diff']**2 + df_[f'y{index}diff']**2)
            df_total_distance.loc[0, str(index)+'dist'] = df_[str(index)+'dist'].sum(axis=0)
        distance_cols = [f'{index}dist' for index in indexes]
        df_total_distance.loc[0, str(key)+'dist'] = df_total_distance[distance_cols].sum(axis=1).values[0]
        y_max = max(y_max, df_total_distance.max().max())
        
    df_total_distance=df_total_distance.div(len(df_))
    dfs.append((patient_id, target_id, df_total_distance))

for j, (patient_id, target_id, df_td_i) in enumerate(dfs):
    for i, key in enumerate(all_keypoints.keys()):
        i_df = df_td_i[[f'{i}dist' for i in all_keypoints[key]]+[f'{key}dist']]
        axes[j, i].bar(i_df.columns, i_df.iloc[0])
        axes[j, i].set_xticklabels(i_df.columns, rotation=90, fontsize=6)
        axes[j, i].set_yscale('log')
        axes[j, i].set_ylim(1e-4, y_max*1.1)
        axes[j, i].grid(True)
        axes[j, i].set_title(f'Patient {patient_id}, {target_id}, {stroke_dict[Y[patient_id-1]]} - {key}')

plt.show()

'''
Patients move less on stroke side - We need to normalize patient size so distances are comparable
For training its important that model knows if exercise is being done by healthy or stroke side
'''
# %%
