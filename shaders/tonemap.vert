#version 450
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 positions[3] = vec2[](
        vec2(-1, -1),
        vec2(3, -1),
        vec2(-1, 3)
    );
    vec2 uv[3] = vec2[](
        vec2(0, 0),
        vec2(2, 0),
        vec2(0, 2)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0, 1);
    v_uv = uv[gl_VertexIndex];
}