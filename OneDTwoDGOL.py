import colour
import cv2
import imutils
import numpy as np
import os
import tqdm
import video_writer

from dataclasses import dataclass, field
from scipy import signal


@dataclass
class GameOfLifeAnimationParams:
    """Keeping track of myriad parameters"""
    output_path: str
    rule: int
    hex_colors: list[str] = field(default_factory=list)
    gol_pcnt: float = 0.5
    video_width: int = 3840
    video_height: int = 2160
    secs: int = 10
    pixel_size: int = 6
    fps: int = 60
    x_offset: int = 0
    random_row_seed: bool = False

    def validate_output_path(self) -> bool:
        return os.path.exists(os.path.dirname(self.output_path))

    def calculate_state_dimensions(self) -> tuple:
        return (self.video_width//self.pixel_size, self.video_height//self.pixel_size)

    def calculate_color_decay_times(self) -> list[int]:
        return [2 * 8 ** i for i in range(len(self.hex_colors) - 1)]

    def generate_color_list(self) -> list:
        color_list = [colour.Color("white")]
        color_decay_times = self.calculate_color_decay_times()
        for i in range(len(self.hex_colors) - 1):
            color_list += list(
                colour.Color(self.hex_colors[i]).range_to(
                    colour.Color(self.hex_colors[i + 1]), color_decay_times[i]
                )
            )
        color_list += [colour.Color("black")]
        return color_list

    def generate_rgb_list(self) -> list:
        color_list = self.generate_color_list()
        return [c.rgb for c in color_list]


## Heavy lifting
class OneDTwoDGameOfLifeAnimation:
    def __init__(self, gol_params) -> None:
        """
        Use the paramter dataclass to instantiate the animator, and play the Game of Life.
        """
        assert gol_params.validate_output_path(), "Provide valid savepath."

        self.rule = gol_params.rule
        self.generate_row_rule_kernel()
        self.video_width, self.video_height = gol_params.video_width, gol_params.video_height
        self.width, self.height = gol_params.calculate_state_dimensions()
        self.gol_height = int(self.height * gol_params.gol_pcnt)
        self.gol_state_width = self.width + 2*self.video_width
        self.gol_state_height = self.gol_height + self.video_height

        self.row_padding = (gol_params.secs*gol_params.fps) // 2
        self.row_width = self.gol_state_width + self.row_padding*2
        self.rows_height = self.height - self.gol_height # to hold all possible rendered 'row' states

        self.row_neighbor_kernel = np.array([1, 2, 4], dtype=np.uint8)
        self.gol_neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # initialize states
        if gol_params.random_row_seed:
            self.row_state = np.random.choice([0, 1], size=self.row_width).astype(np.uint8)
        else:
            self.row_state = np.zeros(self.row_width, np.uint8)
        self.row_state[self.row_width // 2 + gol_params.x_offset] = 1

        self.rows_state = np.concatenate(
            (np.zeros((self.rows_height - 1, self.gol_state_width), np.uint8),
            self.row_state[None, self.row_padding:-self.row_padding]), #<-- quick way to get 1x(len-2) array
        )
        self.gol_state = np.zeros((self.gol_state_height, self.gol_state_width), np.uint8)

        # rendering
        self.colors = (np.array(gol_params.generate_rgb_list(), float) * 255).astype(np.uint8)
        self.decay = np.full((self.height, self.width), len(self.colors) - 1, int)
        self.rgb = None
        self.update_decay()
        self.update_rgb()

    def step(self) -> None:
        self.update_rows_and_gol_state()
        self.update_decay()
        self.update_rgb()

    def update_rgb(self) -> None:
        self.rgb = self.colors[self.decay]

    def update_decay(self) -> None:

        visible_state = np.concatenate(
            (self.gol_state[-self.gol_height:, self.video_width:-self.video_width],
             self.rows_state[:, self.video_width:-self.video_width]),
            axis=0
        )

        self.decay += 1
        self.decay = np.clip(self.decay, None, len(self.colors) - 1)
        self.decay *= 1 - visible_state

    def update_rows_and_gol_state(self) -> None:
        # update rows state, and save "transfer row" that becomes seed for GOL
        rule_idx = signal.convolve2d(self.row_state[None, :], self.row_neighbor_kernel[None, :], mode="same", boundary="wrap")
        self.row_state = self.rule_kernel[rule_idx[0]]
        transfer_row = self.rows_state[:1]
        self.rows_state = np.concatenate((
            self.rows_state[1:],
            self.row_state[None, self.row_padding:-self.row_padding]
        ))

        # update gol
        num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbor_kernel, mode="same", boundary="wrap")
        self.gol_state = np.logical_or(
            num_neighbors == 3,
            np.logical_and(num_neighbors == 2, self.gol_state)
        ).astype(np.uint8)
        self.gol_state = np.concatenate((
            np.zeros((1, self.gol_state_width), np.uint8),
            self.gol_state[1:-1],
            transfer_row
        ))

    def generate_row_rule_kernel(self) -> None:
        self.rule_kernel = np.array([int(x) for x in f'{self.rule:08b}'[::-1]], np.uint8)


def create_video(high_quality_bool=False):
    """
    Video writer
    """
    data = GameOfLifeAnimationParams(
        output_path = "./videos/test.mp4",
        rule = 90,
        secs = 90,
        fps = 30,
        hex_colors = ['#711c91', '#ea00d9', '#0abdc6', '#133e7c', '#091833', '#000103'],
        #hex_colors = ["#FC4D00", "#FCBB00", "#E200B0", "#FF00FF", "#7E00FF", "#0000A2"][::-1],
        random_row_seed = True,
    )
    writer = video_writer.Writer(fps=data.fps, high_quality=high_quality_bool)
    animation = OneDTwoDGameOfLifeAnimation(data)

    for _ in tqdm.trange(data.secs*data.fps):
        small_frame = animation.rgb
        enlarged_frame = imutils.resize(small_frame, data.video_width, data.video_height, cv2.INTER_NEAREST)
        writer.add_frame(enlarged_frame)
        animation.step()
    writer.write(data.output_path)

if __name__ == "__main__":
    create_video(high_quality_bool=True)
