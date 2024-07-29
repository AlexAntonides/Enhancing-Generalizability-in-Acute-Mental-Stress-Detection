import io
import gc

import pywt
import scaleogram as scg

from PIL import Image
import matplotlib.pyplot as plt

def scaleogram(wavelet: str ='morl', window: int =1000 * 10):
    def inner(signal): 
        fig, ax = plt.subplots(figsize=(16, 4))
            
        raise Exception("Doesn't work fully yet")

        scg.cws(
            signal, 
            wavelet=wavelet,
            cbar=None,
            ax=ax,
            clim=(0, 0.005)
        )

        ax.axis("off")
        ax.set_title('')
        fig.set_tight_layout(True)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)

        image = Image.open(img_buf)

        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')
        # Collect data
        gc.collect()
        
        return {
            'image': image
        }
    return inner