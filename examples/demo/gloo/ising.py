
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
//uniform sampler2D u_texture_2;
varying vec2 v_texcoord;
uniform vec2 u_grid_size;

//uniform int u_swap;

vec3 fetch(ivec2 ij) {
    vec2 uv = ij / u_grid_size;
    //if (u_swap < .5)
        return texture2D(u_texture_1, uv).rgb;
    /*else
        return texture2D(u_texture_2, uv).rgb;*/
}

ivec2 grid_pos() {
    return ivec2(round((u_grid_size + 1) * v_texcoord.st));
}

float rand_seed(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

/*float rand() {
    vec2 seed = vec2(0., 0.);
    if (u_swap == 1)
        seed = texture2D(u_texture_1, seed).rg;
    else
        seed = texture2D(u_texture_2, seed).rg;
    return rand_seed(seed);
}*/

float spin(ivec2 ij) {
    float cij = fetch(ij).r;
    if (cij < .5)
        return 0.;
    else
        return 1.;
}

vec4 compute(ivec2 ij) {

    int dx = 0;
    int dy = 0;

    ivec2 ij2 = ivec2(round(u_grid_size)) - ij;
    float rnd = rand_seed(vec2(0., spin(ij2)));

    float cij = spin(ij);

    float kT = 2. / log(1. + sqrt(2.));

    float cnt = 0.;
    for (dx = -1; dx <= 1; dx++) {
        for (dy = -1; (dy <= 1); dy++) {
            if ((dx == 0) && (dy == 0))
                continue;
            cnt += spin(ij + ivec2(dx, dy));
        }
    }
    float dE = 2. * cij * cnt;

    if ((dE <= 0.) || (exp(-dE / kT) > rnd))
        cij = 1. - cij;

    //cij = spin(ij + ivec2(0, 0));
    //cij = 1 - cij;

    return vec4(cij, cij, cij, 1.);
}

void main()
{
    ivec2 ij = grid_pos();
    gl_FragColor = compute(ij);
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

        self.grid_size = self.size[1] // 10, self.size[0] // 10

        tex_shape = self.grid_size + (3,)

        data1 = np.zeros(tex_shape)
        data1[:,:,0] = (np.random.uniform(size=self.grid_size) < .5) * 1

        data1 = (data1 * 255).astype(np.uint8)

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
        # self._program1['u_texture_2'] = self._tex2
        # self._program1['u_swap'] = self._swap
        self._program1['u_grid_size'] = self.grid_size

        self._program2 = gloo.Program(VERT_SHADER1, FRAG_SHADER2)
        self._program2['a_position'] = gloo.VertexBuffer(a_position)
        self._program2['a_texcoord'] = gloo.VertexBuffer(a_tex_coords)
        self._program2['u_texture'] = self._tex2

        self._timer = app.Timer(.1, self.on_timer, start=True)

    def on_timer(self, e):
        self._swap = (1 - self._swap)
        # self._program1['u_swap'] = self._swap
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        fbo = self._fbo2 if self._swap == 0 else self._fbo1
        tex = self._tex1 if self._swap == 0 else self._tex2
        tex2 = self._tex2 if self._swap == 0 else self._tex1

        with fbo:
            gloo.set_clear_color('white')
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, self.grid_size[1], self.grid_size[0])
            self._program1['u_texture_1'] = tex
            self._program1.draw('triangle_strip')

        gloo.set_viewport(0, 0, 1000, 1000)
        gloo.set_clear_color('white')
        gloo.clear(color=True, depth=True)
        self._program2['u_texture'] = tex2
        self._program2.draw('triangle_strip')

if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()
