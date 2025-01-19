#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_tangent;
layout(location = 2) in vec3 v_bitangent;
layout(location = 3) in vec2 v_texcoord;

layout(location = 0) out vec4 f_color;

layout(set = 1, binding = 0) uniform sampler2D u_BaseColorTex;
layout(set = 1, binding = 1) uniform sampler2D u_NormalTex;

layout(set = 0, binding = 1) uniform Light {
    vec3 direction;
    vec3 color;
    vec3 ambient;
} light;

void main() {
    // Fetch the base color from the texture
    vec4 baseColor = texture(u_BaseColorTex, v_texcoord);
    if (baseColor.a < 0.1) discard;

    // Fetch the normal map and transform it to world space
    vec3 normalMap = texture(u_NormalTex, v_texcoord).rgb;
    normalMap = normalMap * 2.0 - 1.0; // Convert from [0, 1] to [-1, 1]

    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 worldNormal = normalize(TBN * normalMap);

    // Compute Lambertian reflectance
    float NdotL = max(dot(worldNormal, -normalize(light.direction)), 0.0);
    vec3 diffuse = light.color * baseColor.rgb * NdotL;

    // Add ambient lighting
    vec3 ambient = light.ambient * baseColor.rgb;

    // Output final color
    f_color = vec4(diffuse + ambient, 1.0);
}