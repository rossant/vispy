
from vispy import gloo
from vispy import app
import numpy as np

a_position = np.array([[-1.0, -1.0, 0.0],
                       [-1.0, +1.0, 0.0],
                       [+1.0, -1.0, 0.0],
                       [+1.0, +1.0, 0.0, ]], np.float32)

a_tex_coords = np.array([[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]], np.float32)

VERT_SHADER1 = """
attribute vec3 a_position;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main (void) {
    v_texcoord = a_texcoord;
    gl_Position = vec4(a_position, 1.0);
}
"""


FRAG_SHADER1 = """
uniform sampler2D u_texture_1;
uniform sampler2D u_texture_2;
varying vec2 v_texcoord;
uniform vec2 u_grid_size;

uniform int u_swap;

vec3 fetch(ivec2 ij) {
    vec2 uv = ij / u_grid_size;
    if (u_swap == 0)
        return texture2D(u_texture_1, uv).rgb;
    else
        return texture2D(u_texture_2, uv).rgb;
}

ivec2 grid_pos() {
    return ivec2(round((u_grid_size + 1) * v_texcoord.st));
}

vec3 compute(ivec2 ij) {
    return fetch(ij + ivec2(1, 0));
}

void main()
{
    ivec2 ij = grid_pos();
    vec3 color = compute(ij);
    gl_FragColor = vec4(color, 1.0);
}

"""


FRAG_SHADER2 = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;

void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
}
"""

class Canvas(app.Canvas):

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')
        self.size = 1000, 1000
        self._swap = 0

        self.grid_size = self.size[1], self.size[0]

        tex_shape = self.grid_size + (3,)
        data1 = np.random.randint(size=tex_shape,
                                  low=0, high=255).astype(np.uint8)

        self._tex1 = gloo.Texture2D(data1)
        self._tex2 = gloo.Texture2D(tex_shape, wrapping='repeat')

        self._fbo1 = gloo.FrameBuffer(self._tex1,
                                      gloo.RenderBuffer(self.grid_size))
        self._fbo2 = gloo.FrameBuffer(self._tex2,
                                      gloo.RenderBuffer(self.grid_size))

        self._program1 = gloo.Program(VERT_SHADER1, FRAG_SHADER1)
        self._program1['a_position'] = gloo.VertexBuffer(a_position)
        self._program1['a_texcoord'] = gloo.VertexBuffer(a_tex_coords)
        self._program1['u_texture_1'] = self._tex1
        self._program1['u_texture_2'] = self._tex2
        self._program1['u_swap'] = self._swap
        self._program1['u_grid_size'] = self.grid_size

        self._program2 = gloo.Program(VERT_SHADER1, FRAG_SHADER2)
        self._program2['a_position'] = gloo.VertexBuffer(a_position)
        self._program2['a_texcoord'] = gloo.VertexBuffer(a_tex_coords)
        self._program2['u_texture'] = self._tex2

        self._timer = app.Timer('auto', self.on_timer, start=True)

    def on_timer(self, e):
        self._swap = (1 - self._swap)
        self._program1['u_swap'] = self._swap
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        fbo = self._fbo2 if self._swap == 0 else self._fbo1
        with fbo:
            gloo.set_clear_color('red')
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, self.grid_size[1], self.grid_size[0])
            self._program1.draw('triangle_strip')

        gloo.set_clear_color('white')
        gloo.clear(color=True, depth=True)
        self._program2.draw('triangle_strip')

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
