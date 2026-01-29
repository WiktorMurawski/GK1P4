#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

const char* TITLE = "GK1P4";

// Globalne zmienne dla kamer
int activeCamera = 0; // 0 = obserwująca, 1 = śledząca, 2 = TPP
int projectionType = 0; // 0 - perspektywiczna, 1 - ortograficzna

// Globalne zmienne dla cyklu dzień/noc
bool dayNightCycleEnabled = true;
float dayNightSpeed = 1.0f;

// Zmienne reflektora
bool spotlightEnabled = true;
float spotlightYaw = 0.0f;
float spotlightPitch = 0.0f;

// Zmienne mgły
bool fogEnabled = true;
float fogDensity = 0.02f;

bool mirrorEnabled = true;

// =====================================================================
//   STRUKTURA MESH - przechowuje geometrię obiektu
// =====================================================================
struct Mesh {
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    unsigned int indexCount;

    Mesh() : VAO(0), VBO(0), EBO(0), indexCount(0) {}
};

// =====================================================================
//   STRUKTURA LIGHT - źródło światła punktowego
// =====================================================================
struct Light {
    glm::vec3 position;
    glm::vec3 diffuse;   // IL - kolor/intensywność światła
    glm::vec3 specular;  // IL - kolor/intensywność dla specular

    // Parametry zanikania (attenuation)
    float constant;
    float linear;
    float quadratic;

    Light(glm::vec3 pos, glm::vec3 col) : 
        position(pos),
        diffuse(col),
        specular(glm::vec3(1.0f)),
        constant(1.0f),
        linear(0.10f),
        quadratic(0.035f)
    { }
};

// =====================================================================
//   STRUKTURA SPOTLIGHT - reflektor
// =====================================================================
struct Spotlight {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 diffuse;
    glm::vec3 specular;

    float cutOff;
    float outerCutOff;

    float constant;
    float linear;
    float quadratic;

    Spotlight() : 
        position(glm::vec3(0.0f)),
        direction(glm::vec3(0.0f, -1.0f, 0.0f)),
        diffuse(glm::vec3(1.0f, 1.0f, 1.0f)),
        specular(glm::vec3(1.0f)),
        cutOff(glm::cos(glm::radians(30.0f))),
        outerCutOff(glm::cos(glm::radians(40.0f))),
        constant(1.0f),
        linear(0.09f),
        quadratic(0.032f)
    { }
};

// =====================================================================
//   SHADERY PHONGA
// =====================================================================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec3 Color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    // Poprawna transformacja normali (transpose inverse)
    Normal = mat3(transpose(inverse(model))) * aNormal;
    Color = aColor;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 Color;

// Właściwości materiału
uniform vec3 viewPos;
uniform float kd;        // kd - współczynnik diffuse (0-1)
uniform float ks;        // ks - współczynnik specular (0-1)
uniform float shininess; // m - wykładnik
uniform float alpha;     // Przezroczystość

// Globalne światło ambient (niezależne od świateł punktowych)
uniform vec3 globalAmbient;

#define MAX_LIGHTS 8
uniform int numLights;

struct Light {
    vec3 position;
    vec3 diffuse;   // IL - kolor/intensywność światła
    vec3 specular;  // IL - kolor/intensywność dla specular
    
    float constant;
    float linear;
    float quadratic;
};

uniform Light lights[MAX_LIGHTS];

// Reflektor (spotlight)
struct Spotlight {
    vec3 position;
    vec3 direction;
    vec3 diffuse;
    vec3 specular;
    
    float cutOff;
    float outerCutOff;
    
    float constant;
    float linear;
    float quadratic;
};

uniform Spotlight spotlight;
uniform bool spotlightEnabled;

uniform bool fogEnabled;
uniform vec3 fogColor;
uniform float fogDensity;

vec3 calculatePointLight(Light light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 objectColor)
{
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Diffuse: kd * IL * IO * (N·L)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = kd * light.diffuse * diff * objectColor;
    
    // Specular: ks * IL * IO * (R·V)^m
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = ks * light.specular * spec * objectColor;
    
    // Zanikanie (attenuation)
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    diffuse *= attenuation;
    specular *= attenuation;
    
    return diffuse + specular;
}

vec3 calculateSpotlight(Spotlight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 objectColor)
{
    vec3 lightDir = normalize(light.position - fragPos);
    
    // Sprawdź czy fragment jest w stożku światła
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = kd * light.diffuse * diff * objectColor;
    
    // Specular
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = ks * light.specular * spec * objectColor;
    
    // Zanikanie
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    
    diffuse *= intensity * attenuation;
    specular *= intensity * attenuation;
    
    return diffuse + specular;
}

void main()
{
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    vec3 ambient = globalAmbient * Color;
    
    vec3 result = vec3(0.0);
    for(int i = 0; i < numLights; i++)
    {
        result += calculatePointLight(lights[i], norm, FragPos, viewDir, Color);
    }

    if (spotlightEnabled)
    {
        result += calculateSpotlight(spotlight, norm, FragPos, viewDir, Color);
    }
    
    result += ambient;

    if (fogEnabled)
    {
        float distance = length(viewPos - FragPos);
        
        float fogFactor = exp(-fogDensity * distance);
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        
        result = mix(fogColor, result, fogFactor);
    }
    
    FragColor = vec4(result, alpha);
    //FragColor = vec4(result, 1.0);
}
)";

// =====================================================================
//   FUNKCJE POMOCNICZE
// =====================================================================
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_1) { activeCamera = 0; std::cout << "Kamera obserwująca" << std::endl; }
        else if (key == GLFW_KEY_2) { activeCamera = 1; std::cout << "Kamera śledząca" << std::endl; }
        else if (key == GLFW_KEY_3) { activeCamera = 2; std::cout << "Kamera TPP" << std::endl; }
        else if (key == GLFW_KEY_P) { projectionType = 0; std::cout << "Rzutowanie perspektywiczne" << std::endl; }
        else if (key == GLFW_KEY_O) { projectionType = 1; std::cout << "Rzutowanie prostokątne" << std::endl; }
        else if (key == GLFW_KEY_N) { dayNightCycleEnabled = !dayNightCycleEnabled; }
        else if (key == GLFW_KEY_EQUAL || key == GLFW_KEY_KP_ADD) { dayNightSpeed += 0.25f; }
        else if (key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) { dayNightSpeed = std::max(0.25f, dayNightSpeed - 0.25f); }
        else if (key == GLFW_KEY_R) { spotlightEnabled = !spotlightEnabled; }
        else if (key == GLFW_KEY_LEFT) { spotlightYaw -= 10.0f; }
        else if (key == GLFW_KEY_RIGHT) { spotlightYaw += 10.0f; }
        else if (key == GLFW_KEY_UP) { spotlightPitch += 10.0f; }
        else if (key == GLFW_KEY_DOWN) { spotlightPitch -= 10.0f; }
        else if (key == GLFW_KEY_M) { fogEnabled = !fogEnabled; }
        else if (key == GLFW_KEY_0) { fogDensity = std::max(0.0f, fogDensity - 0.01f); }
        else if (key == GLFW_KEY_9) { fogDensity = std::min(0.5f, fogDensity + 0.01f); }
        else if (key == GLFW_KEY_L) { mirrorEnabled = !mirrorEnabled; std::cout << "Lustro: " << (mirrorEnabled ? "ON" : "OFF") << std::endl; }
    }
}

unsigned int compileShader(const char* source, GLenum type)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Błąd kompilacji shadera:\n" << infoLog << std::endl;
    }
    return shader;
}

unsigned int createShaderProgram()
{
    unsigned int vs = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    unsigned int fs = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Błąd linkowania programu:\n" << infoLog << std::endl;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

// =====================================================================
//   GENEROWANIE GEOMETRII Z NORMALAMI
// =====================================================================

Mesh createCube()
{
    // Każda ściana ma 4 wierzchołki (pozycja XYZ, normalna XYZ, kolor RGB)
    float vertices[] = {
        // Pozycja           Normalna          Kolor
                // Przód (+Z) - czerwony
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f, 0.0f,

        // Tył (-Z) - zielony
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f,

        // Góra (+Y) - niebieski
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f,

        // Dół (-Y) - żółty
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f, 0.0f,
        
        // Prawa (+X) - cyjan
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f, 1.0f,
        
        // Lewa (-X) - magenta
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f
    };

    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0,      // Przód
        4, 6, 5,  6, 4, 7,      // Tył
        8, 9, 10, 10, 11, 8,    // Góra
        12, 14, 13, 12, 15, 14, // Dół
        16, 18, 17, 16, 19, 18, // Prawa
        20, 21, 22, 20, 22, 23  // Lewa
    };

    Mesh mesh;
    mesh.indexCount = 36;

    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Pozycja
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Normalna
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Kolor
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    return mesh;
}

// Tworzy płaszczyznę (podłoga) z normalną skierowaną w górę
Mesh createPlane()
{
    float vertices[] = {
        // Pozycja              Normalna           Kolor (szary)
        -10.0f, 0.0f, -10.0f,   0.0f, 1.0f, 0.0f,  0.3f, 0.3f, 0.3f,
         10.0f, 0.0f, -10.0f,   0.0f, 1.0f, 0.0f,  0.3f, 0.3f, 0.3f,
         10.0f, 0.0f,  10.0f,   0.0f, 1.0f, 0.0f,  0.3f, 0.3f, 0.3f,
        -10.0f, 0.0f,  10.0f,   0.0f, 1.0f, 0.0f,  0.3f, 0.3f, 0.3f
    };

    unsigned int indices[] = {
        0, 2, 1,
        2, 0, 3
    };

    Mesh mesh;
    mesh.indexCount = 6;

    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    return mesh;
}

Mesh createMirror()
{
    float vertices[] = {
        -0.5f,-0.5f,0.0f, 0.0f,0.0f,1.0f, 0.8f,0.8f,0.9f,
         0.5f,-0.5f,0.0f, 0.0f,0.0f,1.0f, 0.8f,0.8f,0.9f,
         0.5f, 0.5f,0.0f, 0.0f,0.0f,1.0f, 0.8f,0.8f,0.9f,
        -0.5f, 0.5f,0.0f, 0.0f,0.0f,1.0f, 0.8f,0.8f,0.9f
    };
    unsigned int indices[] = { 0,1,2, 2,3,0 };

    Mesh mesh;
    mesh.indexCount = 6;
    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);
    glBindVertexArray(mesh.VAO);
    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
    return mesh;
}

// Tworzy kulę metodą UV sphere
Mesh createSphere(int stacks = 30, int slices = 30)
{
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // Generowanie wierzchołków
    for (int i = 0; i <= stacks; ++i)
    {
        float phi = M_PI * float(i) / float(stacks); // 0 do PI

        for (int j = 0; j <= slices; ++j)
        {
            float theta = 2.0f * M_PI * float(j) / float(slices); // 0 do 2*PI

            // Pozycja na sferze jednostkowej
            float x = sin(phi) * cos(theta);
            float y = cos(phi);
            float z = sin(phi) * sin(theta);

            // Normalna = znormalizowana pozycja
            float nx = x;
            float ny = y;
            float nz = z;

            // Kolor zależny od pozycji (gradient)
            float r = (x + 1.0f) * 0.5f;
            float g = (y + 1.0f) * 0.5f;
            float b = (z + 1.0f) * 0.5f;

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
            vertices.push_back(r);
            vertices.push_back(g);
            vertices.push_back(b);
        }
    }

    // Generowanie indeksów (trójkąty)
    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            int first = i * (slices + 1) + j;
            int second = first + slices + 1;

            // Pierwszy trójkąt
            indices.push_back(first);
            indices.push_back(first + 1);
            indices.push_back(second);

            // Drugi trójkąt
            indices.push_back(second + 1);
            indices.push_back(second);
            indices.push_back(first + 1);
        }
    }

    Mesh mesh;
    mesh.indexCount = indices.size();

    glGenVertexArrays(1, &mesh.VAO);
    glGenBuffers(1, &mesh.VBO);
    glGenBuffers(1, &mesh.EBO);

    glBindVertexArray(mesh.VAO);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);

    return mesh;
}

// Funkcja rysująca mesh
void drawMesh(const Mesh& mesh, unsigned int shaderProgram, const glm::mat4& model)
{
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glBindVertexArray(mesh.VAO);
    glDrawElements(GL_TRIANGLES, mesh.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

// Funkcja przekazująca światła do shadera
void setLights(unsigned int shaderProgram, const std::vector<Light>& lights)
{
    glUniform1i(glGetUniformLocation(shaderProgram, "numLights"), lights.size());

    for (size_t i = 0; i < lights.size(); ++i)
    {
        std::string base = "lights[" + std::to_string(i) + "]";

        glUniform3fv(glGetUniformLocation(shaderProgram, (base + ".position").c_str()), 1, glm::value_ptr(lights[i].position));
        glUniform3fv(glGetUniformLocation(shaderProgram, (base + ".diffuse").c_str()), 1, glm::value_ptr(lights[i].diffuse));
        glUniform3fv(glGetUniformLocation(shaderProgram, (base + ".specular").c_str()), 1, glm::value_ptr(lights[i].specular));

        glUniform1f(glGetUniformLocation(shaderProgram, (base + ".constant").c_str()), lights[i].constant);
        glUniform1f(glGetUniformLocation(shaderProgram, (base + ".linear").c_str()), lights[i].linear);
        glUniform1f(glGetUniformLocation(shaderProgram, (base + ".quadratic").c_str()), lights[i].quadratic);
    }
}

// Funkcja przekazująca reflektor do shadera
void setSpotlight(unsigned int shaderProgram, const Spotlight& spot)
{
    glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.position"), 1, glm::value_ptr(spot.position));
    glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.direction"), 1, glm::value_ptr(spot.direction));
    glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.diffuse"), 1, glm::value_ptr(spot.diffuse));
    glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.specular"), 1, glm::value_ptr(spot.specular));
    
    glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.cutOff"), spot.cutOff);
    glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.outerCutOff"), spot.outerCutOff);
    
    glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.constant"), spot.constant);
    glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.linear"), spot.linear);
    glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.quadratic"), spot.quadratic);
}

// =====================================================================
//   MAIN
// =====================================================================
int main()
{
    // ==================== Inicjalizacja GLFW + GLAD ====================
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    GLFWwindow* window = glfwCreateWindow(1200, 900, TITLE, nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Nie udało się utworzyć okna GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);

    if (!gladLoadGL()) {
        std::cerr << "Nie udało się zainicjalizować GLAD / OpenGL" << std::endl;
        return -1;
    }

    int major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "OpenGL " << major << "." << minor << std::endl;

    // ==================== Shadery ====================
    unsigned int shaderProgram = createShaderProgram();

    // ==================== Tworzenie obiektów ====================
    std::cout << "Generowanie meshy..." << std::endl;

    Mesh cubeMesh = createCube();
    Mesh planeMesh = createPlane();
    Mesh sphereMesh = createSphere(30, 30);
    Mesh mirrorMesh = createMirror();

    std::cout << "\n=== STEROWANIE ===" << std::endl;
    std::cout << "ESC - Wyjście\n" << std::endl;
    std::cout << "1 - Kamera obserwująca (statyczna)" << std::endl;
    std::cout << "2 - Kamera śledząca obiekt" << std::endl;
    std::cout << "3 - Kamera TPP (Third Person)" << std::endl;
    std::cout << "N - Włącz/wyłącz cykl dzień/noc" << std::endl;
    std::cout << "R - Włącz/wyłącz reflektor" << std::endl;
    std::cout << "Strzałki - Kierunek reflektora" << std::endl;
    std::cout << "+ - Przyspiesz cykl dzień/noc" << std::endl;
    std::cout << "- - Zwolnij cykl dzień/noc" << std::endl;
    std::cout << "P - rzutowanie perspektywiczne" << std::endl;
    std::cout << "O - rzutowanie ortogonalne" << std::endl;
    std::cout << "M - Włącz/wyłącz mgłę" << std::endl;
    std::cout << "9 - Zmniejsz gęstość mgły" << std::endl;
    std::cout << "0 - Zwiększ gęstość mgły" << std::endl;

    // ==================== Tworzenie świateł ====================
    std::vector<Light> lights;

    // Światło 1: Białe nad sceną (główne)
    lights.push_back(Light(glm::vec3(0.0f, 8.0f, 0.0f), glm::vec3(0.8f, 0.8f, 0.8f)));

    // Światło 2: Czerwone z boku
    lights.push_back(Light(glm::vec3(5.0f, 3.0f, 5.0f), glm::vec3(0.8f, 0.2f, 0.2f)));

    // Światło 3: Niebieskie z drugiego boku
    lights.push_back(Light(glm::vec3(-1.0f, 3.0f, -2.0f), glm::vec3(0.2f, 0.2f, 0.8f)));

    // Światło 4: W środku sceny
    lights.push_back(Light(glm::vec3(0.0f, 0.1f, 0.0f), glm::vec3(0.3f, 0.3f, 0.3f)));

    std::cout << "\nDodano " << lights.size() << " świateł punktowych" << std::endl;


    // ==================== Inicjalizacja reflektora ====================
    Spotlight spotlight;
    std::cout << "Dodano reflektor na ruchomym obiekcie" << std::endl;

    // ==================== Ustawienia OpenGL ====================
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_STENCIL_TEST);

    // ==================== Pętla główna ====================
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        float time = (float)glfwGetTime();

        // Kolor tła zmienia się z cyklem dzień/noc
        glm::vec3 skyColor;

        if (dayNightCycleEnabled)
        {
            float cycle = sin(time*dayNightSpeed);
            float dayness = (cycle + 1.0f) * 0.5f;  // 0 = noc, 1 = dzień

            glm::vec3 nightSky = glm::vec3(0.02f, 0.02f, 0.08f);
            glm::vec3 daySky = glm::vec3(0.4f, 0.6f, 0.9f);

            skyColor = glm::mix(nightSky, daySky, dayness);
        }
        else
        {
            skyColor = glm::vec3(0.1f, 0.1f, 0.1f); 
        }

        glm::vec3 fogColor = glm::vec3(0.1f, 0.2f, 0.3f);

        glClearColor(skyColor.r, skyColor.g, skyColor.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // ==================== OBLICZANIE POZYCJI RUCHOMEGO OBIEKTU ====================
        float radius = 5.0f;
        float speed = 0.3f;
        float movingX = radius * cos(time * speed);
        float movingZ = radius * sin(time * speed);
        float movingY = 0.0f;
        glm::vec3 movingObjPosition(movingX, movingY, movingZ);

        // Kierunek ruchu obiektu (do kamery TPP)
        glm::vec3 movingDirection = glm::normalize(glm::vec3(-sin(time * speed), 0.0f, cos(time * speed)));

        // ==================== MACIERZE ====================

        // Pozycja kamery
        glm::vec3 cameraPosition;

        // Macierz View - zależna od kamery
        glm::mat4 view;

        if (activeCamera == 0)
        {
            // KAMERA 0: Obserwująca
            cameraPosition = glm::vec3(5.0f, 5.0f, 10.0f);
            view = glm::lookAt(
                cameraPosition,
                glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f)
            );
        }
        else if (activeCamera == 1)
        {
            // KAMERA 1: Śledząca
            cameraPosition = glm::vec3(8.0f, 6.0f, 8.0f);
            view = glm::lookAt(
                cameraPosition,
                movingObjPosition,
                glm::vec3(0.0f, 1.0f, 0.0f)
            );
        }
        else if (activeCamera == 2)
        {
            // KAMERA 2: TPP
            float cameraDistance = 3.0f;
            float cameraHeight = 2.0f;

            cameraPosition = movingObjPosition - movingDirection * cameraDistance;
            cameraPosition.y += cameraHeight;

            view = glm::lookAt(
                cameraPosition,
                movingObjPosition,
                glm::vec3(0.0f, 1.0f, 0.0f)
            );
        }

        // Rzutowanie perspektywiczne
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glm::mat4 projection;
        if (projectionType == 0) {
            projection = glm::perspective(
                glm::radians(45.0f),
                (float)width / (float)height,
                0.1f, 100.0f
            );
        }
        else if (projectionType == 1) {
            float aspect = (float)width / (float)height;
            float orthoHeight = 10.0f;
            float orthoWidth = orthoHeight * aspect;

            projection = glm::ortho(
                -orthoWidth, orthoWidth,
                -orthoHeight, orthoHeight,
                0.1f, 100.0f
            );
        }

        // ==================== PRZEKAZANIE UNIFORMÓW ====================
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPosition));

        // ==================== CYKL DZIEŃ/NOC ====================
        glm::vec3 globalAmbient;

        if (dayNightCycleEnabled)
        {
            float cycle = sin(time * dayNightSpeed);

            float ambientStrength = 0.20f + 0.15f * cycle;

            float dayness = (cycle + 1.0f) * 0.5f;

            glm::vec3 nightColor = glm::vec3(0.8f, 0.8f, 1.2f);
            glm::vec3 dayColor = glm::vec3(1.0f, 1.0f, 0.9f);

            glm::vec3 ambientColor = glm::mix(nightColor, dayColor, dayness);
            globalAmbient = ambientStrength * ambientColor;
        }
        else
        {
            globalAmbient = glm::vec3(0.15f, 0.15f, 0.2f);
        }

        glUniform3fv(glGetUniformLocation(shaderProgram, "globalAmbient"), 1, glm::value_ptr(globalAmbient));

        // ==================== AKTUALIZACJA REFLEKTORA ====================
        // Reflektor jest umieszczony na ruchomym obiekcie
        spotlight.position = movingObjPosition;

        // Oblicz kierunek reflektora na podstawie yaw i pitch
        glm::vec3 direction;
        direction.x = cos(glm::radians(spotlightYaw)) * cos(glm::radians(spotlightPitch));
        direction.y = sin(glm::radians(spotlightPitch));
        direction.z = sin(glm::radians(spotlightYaw)) * cos(glm::radians(spotlightPitch));
        spotlight.direction = glm::normalize(direction);

        // Przekazanie reflektora do shadera
        setSpotlight(shaderProgram, spotlight);
        glUniform1i(glGetUniformLocation(shaderProgram, "spotlightEnabled"), spotlightEnabled);

        // Przekazanie świateł
        setLights(shaderProgram, lights);

        // Mgła
        glUniform1i(glGetUniformLocation(shaderProgram, "fogEnabled"), fogEnabled);
        glUniform3fv(glGetUniformLocation(shaderProgram, "fogColor"), 1, glm::value_ptr(fogColor));
        glUniform1f(glGetUniformLocation(shaderProgram, "fogDensity"), fogDensity);

        // ==================== RYSOWANIE OBIEKTÓW ====================

        // Macierz lustra
        glm::mat4 mirrorModel = glm::translate(glm::mat4(1.0f), movingObjPosition);
        mirrorModel = glm::rotate(mirrorModel, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
        mirrorModel = glm::scale(mirrorModel, glm::vec3(1.5f, 1.5f, 0.1f));

        if (mirrorEnabled)
        {
            // PASS 1: Stencil
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            glStencilMask(0xFF);
            glDepthMask(GL_FALSE);
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

            glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 1.0f);
            drawMesh(mirrorMesh, shaderProgram, mirrorModel);

            // NEW: Clear depth buffer ONLY in the mirror region (where stencil == 1)
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilMask(0x00);  // Don't modify stencil
            glDepthMask(GL_TRUE);  // Allow depth writes
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);  // Don't modify color

            glClear(GL_DEPTH_BUFFER_BIT);

            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);  // Re-enable color writes

            // PASS 2: Reflection
            glStencilFunc(GL_EQUAL, 1, 0xFF);
            glStencilMask(0x00);
            glDepthMask(GL_TRUE);
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            glm::mat4 reflectionMatrix = glm::mat4(1.0f);
            reflectionMatrix = glm::translate(reflectionMatrix, movingObjPosition);
            reflectionMatrix = glm::rotate(reflectionMatrix, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
            reflectionMatrix = glm::scale(reflectionMatrix, glm::vec3(1.0f, 1.0f, -1.0f));
            reflectionMatrix = glm::rotate(reflectionMatrix, -time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
            reflectionMatrix = glm::translate(reflectionMatrix, -movingObjPosition);

            glCullFace(GL_FRONT);
            glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 1.0f);

            glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.8f);
            glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.1f);
            glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 1.0f);
            drawMesh(planeMesh, shaderProgram, reflectionMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -2.0f, 0.0f)));

            glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);
            glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
            glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);
            glm::mat4 ms = reflectionMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(-3.0f, 0.5f, 0.0f));
            drawMesh(sphereMesh, shaderProgram, glm::scale(ms, glm::vec3(1.5f)));

            glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);
            glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
            glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);
            glm::mat4 mc1 = reflectionMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.5f, -2.0f));
            drawMesh(cubeMesh, shaderProgram, glm::rotate(mc1, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f)));

            glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.9f);
            glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.05f);
            glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 5.0f);
            glm::mat4 mc2 = reflectionMatrix * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.5f, -4.0f));
            drawMesh(cubeMesh, shaderProgram, glm::scale(mc2, glm::vec3(0.8f, 1.5f, 0.8f)));

            glCullFace(GL_BACK);

            // PASS 3: Mirror surface
            glStencilFunc(GL_ALWAYS, 0, 0xFF);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);  // ADD THIS LINE
            glStencilMask(0xFF);  // CHANGE from 0x00 to 0xFF

            //glDepthFunc(GL_ALWAYS);  // ADD THIS - always pass depth test
            //glDepthMask(GL_FALSE);   // ADD THIS - but don't write to depth buffer

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.3f);
            glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
            glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);
            glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 0.3f);
            drawMesh(mirrorMesh, shaderProgram, mirrorModel);

            glDisable(GL_BLEND);
        }

        // PASS 4: Normal scene
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilMask(0x00);
        glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 1.0f);

        // 1. PODŁOGA (statyczna)
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.8f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.1f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 1.0f);

        glm::mat4 modelFloor = glm::mat4(1.0f);
        modelFloor = glm::translate(modelFloor, glm::vec3(0.0f, -2.0f, 0.0f));
        drawMesh(planeMesh, shaderProgram, modelFloor);

        // 2. KULA (statyczna, gładki obiekt)
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);

        glm::mat4 modelSphere = glm::mat4(1.0f);
        modelSphere = glm::translate(modelSphere, glm::vec3(-3.0f, 0.5f, 0.0f));
        modelSphere = glm::scale(modelSphere, glm::vec3(1.5f, 1.5f, 1.5f));
        drawMesh(sphereMesh, shaderProgram, modelSphere);

        // 3. SZEŚCIAN STATYCZNY
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);

        glm::mat4 modelCube1 = glm::mat4(1.0f);
        modelCube1 = glm::translate(modelCube1, glm::vec3(3.0f, 0.5f, -2.0f));
        modelCube1 = glm::rotate(modelCube1, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        drawMesh(cubeMesh, shaderProgram, modelCube1);

        // 4. SZEŚCIAN STATYCZNY
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.9f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.05f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 5.0f);

        glm::mat4 modelCube2 = glm::mat4(1.0f);
        modelCube2 = glm::translate(modelCube2, glm::vec3(0.0f, 0.5f, -4.0f));
        modelCube2 = glm::scale(modelCube2, glm::vec3(0.8f, 1.5f, 0.8f));
        drawMesh(cubeMesh, shaderProgram, modelCube2);

        // 5. SZEŚCIAN RUCHOMY
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.7f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.6f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 64.0f);

        glm::mat4 modelMoving = glm::mat4(1.0f);
        modelMoving = glm::translate(modelMoving, movingObjPosition);
        modelMoving = glm::rotate(modelMoving, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
        modelMoving = glm::rotate(modelMoving, time * 0.5f, glm::vec3(1.0f, 0.0f, 0.0f));
        modelMoving = glm::scale(modelMoving, glm::vec3(0.7f, 0.7f, 0.7f));
        drawMesh(cubeMesh, shaderProgram, modelMoving);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ==================== Sprzątanie ====================
    glDeleteVertexArrays(1, &cubeMesh.VAO);
    glDeleteBuffers(1, &cubeMesh.VBO);
    glDeleteBuffers(1, &cubeMesh.EBO);

    glDeleteVertexArrays(1, &planeMesh.VAO);
    glDeleteBuffers(1, &planeMesh.VBO);
    glDeleteBuffers(1, &planeMesh.EBO);

    glDeleteVertexArrays(1, &sphereMesh.VAO);
    glDeleteBuffers(1, &sphereMesh.VBO);
    glDeleteBuffers(1, &sphereMesh.EBO);

    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}