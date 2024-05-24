import numpy as np
import matplotlib.pyplot as plt
from datetime import date

def figtitle(title, subtitle=None, fignames="auto", fignum=None, draft=True):

    # figure title
    suptitle = plt.suptitle(f"Fig. {fignum} | {title}" if fignum else title, ha="left", va="bottom", x=0.125, y=0.95, size="x-large", weight="bold")

    # subplot titles
    if draft:
        annotation = date.today().strftime("%d.%m.%Y") + " DRAFT"
        plt.text(0.9, suptitle._y, annotation, transform=plt.gcf().transFigure, ha="right", va="bottom", style="italic", color="red")

    # subplot titles
    axs = np.ravel(plt.gcf().get_axes())

    if fignames == "auto":
        if len(axs) > 0:
            fignames = [chr(_) for _ in  range(65, 65+len(axs))]
        else:
            fignames = [fignames]

    if subtitle is None:
        subtitle = ["" for _ in fignames]
    else:
        if len(axs) == 1:
            subtitle = [subtitle]
        
        
    for i in range(len(axs)):
        axs[i].set_title(r"$\bf{" + fignames[i] + "}$" + " " + subtitle[i], size="large")

plt.figtitle = figtitle