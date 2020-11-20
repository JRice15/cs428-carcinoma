import math
import re
import os

import matplotlib.pyplot as plt
import numpy as np



def get_num_str(num):
    if num == 0:
        return "0.0"
    if num < 0.001 or num > 999:
        num = "{:.2E}".format(num)
    else:
        num = "{:.5f}".format(num)
    return num

def save_history(H, run_name, secs, settings, marker_step=1, skip=0):
    """
    Args:
        H: keras.callbacks.History object
        settings: Settings object
    """
    os.makedirs("stats/"+run_name, exist_ok=True)
    epochs_ran = len(H.history["loss"])
    statsfile_name = "stats/" + run_name + "/stats.txt"
    with open(statsfile_name, "w") as f:
        f.write(run_name + "\n\n")
        f.write("Epochs ran:\t\t\t{}\n".format(epochs_ran))
        f.write("Secs per epoch:\t\t{}\n".format(secs / epochs_ran))
        f.write("Minutes total:\t\t{}\n".format(secs / 60))
        f.write("Hours total:\t\t{}\n".format(secs / 3600))
    settings.write_to_file(statsfile_name)
    for k in H.history.keys():
        if not k.startswith("val_"):
            # skips first couple epochs for clearer scale
            if len(H.history[k]) < 2 * skip:
                skip = 0
            train_data = H.history[k][skip:]
            try:
                valdata = H.history["val_"+k][skip:]
                mark = 1
            except KeyError:
                valdata = None
                mark = 0
            data = (train_data, valdata)
            xrange = list(range(skip, len(train_data)+skip))
            make_plot(xrange=xrange, data=data, axlabels=(run_name,k), mark=mark,
                dnames=("train","validation"), title=k+" by epoch", marker_step=marker_step,
                skipshift=skip, directory="stats/"+run_name, filename=k+".png")

            with open(statsfile_name, "a") as f:
                if valdata is None:
                    f.write("Final {}:\t\t\t{}\n".format(k, train_data[-1]))
                    f.write("Min {}:\t\t\t{}\n".format(k, min(train_data)))
                else:
                    f.write("Final {}:\t\t\t{}\tval:\t{}\n".format(k, train_data[-1], valdata[-1]))
                    f.write("Min {}:\t\t\t{}\tval:\t{}\n".format(k, min(train_data), min(valdata)))



def make_plot(xrange, data, title, axlabels, directory, filename, dnames=None, marker_step=1, 
        mark=0, legendloc="upper right", skipshift=0, fillbetweens=None, 
        fillbetween_desc="", ylim=None, ymin=None):
    """
    make a pretty matplotlib plot
    Args:
        data: tuple of lists/arrays, each of which is a data line
        mark: index of data to mark with value labels
        dnames: tuple of data names
        axlabels: tuple (xname, yname)
    """
    assert isinstance(data, tuple)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    global_min = 1e20
    global_max = -1e20
    mark_data = None
    for i in range(len(data)):
        if data[i] is None:
            continue
        if i == mark:
            mark_data = data[i]
        global_min = min(np.min(data[i]), global_min)
        global_max = max(np.max(data[i]), global_max)
        plt.plot(xrange, data[i])
        if fillbetweens is not None:
            plt.fill_between(xrange, fillbetweens[i][0], fillbetweens[i][1], alpha=0.15)

    if mark_data is not None:
        mark_indices = list(range(xrange[0],xrange[-1],marker_step))
        points = np.interp(mark_indices, xrange, mark_data)
        # points = mark_data[mark_indices]
        up = True
        for i,y in enumerate(points):
            valstr = get_num_str(y)
            if up:
                xytext = (0,5)
                up = False
            else:
                xytext = (0,-12)
                up = True
            xy = (xrange[0] + marker_step*i, y)
            plt.plot(*xy, marker=".", mfc="black", mec="black", markersize=5)
            plt.annotate(valstr, xy=xy, xytext=xytext, 
                horizontalalignment="center", textcoords="offset points")
    
        valstr = get_num_str(mark_data[-1])
        ytext = 5 if up else -12
        plt.annotate(valstr, xy=(xrange[-1], mark_data[-1]), xytext=(-7,ytext), textcoords="offset points")
        plt.plot(xrange[-1], mark_data[-1], marker=".", color="green")

    plt.title(title)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1] + " " + fillbetween_desc)
    plt.tight_layout()
    # plt.yscale("log")
    if dnames is not None:
        dnames = [i for i in dnames if i is not None]
        plt.legend(dnames, loc=legendloc)

    current_bot, current_top = plt.ylim()
    if ylim is not None:
        current_top = min(ylim, current_top)
    global_min = global_min - (0.03 * abs(current_top - global_min))
    bot_diff = 0
    if ymin is not None:
        global_min = min(ymin, global_min, current_bot)
        bot_diff = max(current_bot - global_min, 0)
    new_top = current_top + (0.1 * bot_diff)
    plt.ylim(bottom=global_min, top=new_top)

    plt.margins(x=0.125, y=0.1)

    fname = os.path.join(directory, filename)
    plt.savefig(fname, dpi=300)
    plt.close()