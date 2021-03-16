import sys
import matplotlib.pyplot as plt

class CropInterface(object):
    """
    Like Cursor but the crosshair snaps to the nearest x, y point.
    For simplicity, this assumes that *x* is sorted.
    """

    # def __init__(self, ax, x, y):
    def __init__(self, axs, ims, default_border = [-1, -1, -1, -1]):
        assert len(axs) == len(ims)
        self.axs = axs
        self.ims = ims
        self.click_counter = 0

        assert len(default_border) == 4
        self.border = []
        for border, default in zip(default_border, [0, 0, ims[0].size[0], ims[0].size[1]]):
            self.border.append(border if border != -1 else default)
        
        self.img_update()

    def img_update_crop0(self, x, y):
        max_width = self.border[2] - self.border[0]
        max_length = self.border[3] - self.border[1]
        width, length = max_width - x, max_length - y
        wlmin = min(width, length)
        self.border[0] += x
        self.border[2] = self.border[0] + wlmin
        self.border[1] += y
        self.border[3] = self.border[1] + wlmin
        self.img_update()

    def img_update_crop1(self, x, y):
        wlmin = min(x, y)
        self.border[0] += x - wlmin
        self.border[2] = self.border[0] + wlmin
        self.border[1] += y - wlmin
        self.border[3] = self.border[1] + wlmin
        self.img_update()

    def img_update(self):
        print([int(i) for i in self.border])
        for i, ax in enumerate(self.axs):
            ax.imshow(self.ims[i].crop(self.border))
        plt.draw()

    def on_click(self, event):
        x, y = event.xdata, event.ydata
        if x is not None:
            self.click_counter = (self.click_counter + 1) % 2
            if self.click_counter == 1:
                self.img_update_crop0(x, y)
            else:
                self.img_update_crop1(x, y)

    def on_press(self, event):
        if event.key == 'escape':
            print("Reset")
            self.click_counter = 0
            self.border = [0, 0, self.ims[0].size[0], self.ims[0].size[1]]
            self.img_update_crop0(0,0)
        sys.stdout.flush()

