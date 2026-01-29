#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#define M_PI 3.14159265358979323846

const char* TITLE = "GK1P4 - Faza 3: Oświetlenie Phonga (Global Ambient)";

// =====================================================================
//   GLOBALNE ZMIENNE DLA KAMER
// =====================================================================
int activeCamera = 0; // 0 = obserwująca, 1 = śledząca, 2 = TPP
int projectionType = 0; // 0 - perspektywiczna, 1 - ortograficzna

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

    Light(glm::vec3 pos, glm::vec3 col)
        : position(pos),
        diffuse(col),
        specular(glm::vec3(1.0f)),
        constant(1.0f),
        linear(0.10f),
        quadratic(0.035f)
    {
    }
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
uniform float kd;        // Współczynnik diffuse (0-1)
uniform float ks;        // Współczynnik specular (0-1)
uniform float shininess; // m - wykładnik (1-128)

// Globalne światło ambient (niezależne od świateł punktowych)
uniform vec3 globalAmbient;

// Maksymalnie 4 światła
#define MAX_LIGHTS 4
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
    
    return diffuse + specular;  // BEZ ambient!
}

void main()
{
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);
    
    // Globalne światło ambient (raz dla całej sceny)
    vec3 ambient = globalAmbient * Color;
    
    // Suma wszystkich świateł punktowych (diffuse + specular)
    vec3 result = vec3(0.0);
    for(int i = 0; i < numLights; i++)
    {
        result += calculatePointLight(lights[i], norm, FragPos, viewDir, Color);
    }
    
    // Dodaj ambient na koniec (raz!)
    result += ambient;
    
    FragColor = vec4(result, 1.0);
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
        if (key == GLFW_KEY_1)
        {
            activeCamera = 0;
            std::cout << "Kamera: OBSERWUJĄCA (statyczna)" << std::endl;
        }
        else if (key == GLFW_KEY_2)
        {
            activeCamera = 1;
            std::cout << "Kamera: ŚLEDZĄCA obiekt (statyczna pozycja)" << std::endl;
        }
        else if (key == GLFW_KEY_3)
        {
            activeCamera = 2;
            std::cout << "Kamera: TPP (Third Person - podąża za obiektem)" << std::endl;
        }
        else if (key == GLFW_KEY_P)
        {
            projectionType = 0;
            std::cout << "Rzutowanie: PERSPEKTYWICZNE" << std::endl;
        }
        else if (key == GLFW_KEY_O)
        {
            projectionType = 1;
            std::cout << "Rzutowanie: ORTOGONALNE" << std::endl;
        }
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

// Tworzy sześcian z poprawnymi normalami (każda ściana ma własne wierzchołki)
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

// Tworzy kulę metodą UV sphere - normalna = pozycja (dla sfery jednostkowej)
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
    std::cout << "Generowanie geometrii..." << std::endl;

    Mesh cubeMesh = createCube();
    Mesh planeMesh = createPlane();
    Mesh sphereMesh = createSphere(30, 30);

    std::cout << "Gotowe!" << std::endl;
    std::cout << "\n=== STEROWANIE ===" << std::endl;
    std::cout << "1 - Kamera obserwująca (statyczna)" << std::endl;
    std::cout << "2 - Kamera śledząca obiekt" << std::endl;
    std::cout << "3 - Kamera TPP (Third Person)" << std::endl;
    std::cout << "ESC - Wyjście\n" << std::endl;
    std::cout << "Aktywna kamera: OBSERWUJĄCA (statyczna)" << std::endl;

    // ==================== Tworzenie świateł ====================
    std::vector<Light> lights;

    // Światło 1: Białe nad sceną (główne)
    lights.push_back(Light(glm::vec3(0.0f, 8.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f)));

    // Światło 2: Czerwone z boku
    lights.push_back(Light(glm::vec3(5.0f, 3.0f, 5.0f), glm::vec3(1.0f, 0.3f, 0.3f)));

    // Światło 3: Niebieskie z drugiego boku
    lights.push_back(Light(glm::vec3(-1.0f, 3.0f, -2.0f), glm::vec3(0.3f, 0.3f, 1.0f)));

    // Światło 4: W środku sceny
    lights.push_back(Light(glm::vec3(0.0f, 0.1f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f)));

    std::cout << "\nDodano " << lights.size() << " świateł punktowych" << std::endl;

    // ==================== Ustawienia OpenGL ====================
    glEnable(GL_DEPTH_TEST);
    // Face culling wyłączony na razie - włączymy później jeśli będzie potrzebny
     glEnable(GL_CULL_FACE);

    // ==================== Pętla główna ====================
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        float time = (float)glfwGetTime();

        // ==================== OBLICZANIE POZYCJI RUCHOMEGO OBIEKTU ====================
        float radius = 4.0f;
        float speed = 0.5f;
        float movingX = radius * cos(time * speed);
        float movingZ = radius * sin(time * speed);
        float movingY = 1.0f;
        glm::vec3 movingObjPosition(movingX, movingY, movingZ);

        // Kierunek ruchu obiektu (do kamery TPP)
        glm::vec3 movingDirection = glm::normalize(glm::vec3(-sin(time * speed), 0.0f, cos(time * speed)));

        // ==================== MACIERZE ====================

        // Pozycja kamery (potrzebna do speculara w Phongu)
        glm::vec3 cameraPosition;

        // MACIERZ VIEW - w zależności od aktywnej kamery
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
            projection = glm::ortho(
                -10.0f, 10.0f,
                -10.0f, 10.0f,
                0.1f, 100.0f
            );
        }



        // ==================== PRZEKAZANIE UNIFORMÓW ====================
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, glm::value_ptr(cameraPosition));

        // Globalne światło ambient (niezależne od liczby świateł punktowych)
        glm::vec3 globalAmbient = glm::vec3(0.2f, 0.2f, 0.2f);  // Słaby ambient
        glUniform3fv(glGetUniformLocation(shaderProgram, "globalAmbient"), 1, glm::value_ptr(globalAmbient));

        // Przekazanie świateł
        setLights(shaderProgram, lights);

        // ==================== RYSOWANIE OBIEKTÓW ====================

        // 1. PODŁOGA (statyczna) - matowa, szara
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.8f);        // Dużo diffuse
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.1f);        // Mało specular (matowa)
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 8.0f); // Niski wykładnik

        glm::mat4 modelFloor = glm::mat4(1.0f);
        modelFloor = glm::translate(modelFloor, glm::vec3(0.0f, -2.0f, 0.0f));
        drawMesh(planeMesh, shaderProgram, modelFloor);

        // 2. KULA (statyczna, gładki obiekt) - bardzo błyszcząca, metaliczna
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);         // Średni diffuse
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);         // Dużo specular!
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f); // Wysoki wykładnik (ostry odblask)

        glm::mat4 modelSphere = glm::mat4(1.0f);
        modelSphere = glm::translate(modelSphere, glm::vec3(-3.0f, 0.5f, 0.0f));
        modelSphere = glm::scale(modelSphere, glm::vec3(1.5f, 1.5f, 1.5f));
        drawMesh(sphereMesh, shaderProgram, modelSphere);

        // 3. SZEŚCIAN STATYCZNY #1 - plastikowy
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.6f);
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.9f);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 128.0f);

        glm::mat4 modelCube1 = glm::mat4(1.0f);
        modelCube1 = glm::translate(modelCube1, glm::vec3(3.0f, 0.5f, -2.0f));
        modelCube1 = glm::rotate(modelCube1, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        drawMesh(cubeMesh, shaderProgram, modelCube1);

        // 4. SZEŚCIAN STATYCZNY #2 - gumowy (bardzo matowy)
        glUniform1f(glGetUniformLocation(shaderProgram, "kd"), 0.9f);        // Dużo diffuse
        glUniform1f(glGetUniformLocation(shaderProgram, "ks"), 0.05f);       // Prawie brak specular
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), 5.0f); // Bardzo niski (rozmazany)

        glm::mat4 modelCube2 = glm::mat4(1.0f);
        modelCube2 = glm::translate(modelCube2, glm::vec3(0.0f, 0.5f, -4.0f));
        modelCube2 = glm::scale(modelCube2, glm::vec3(0.8f, 1.5f, 0.8f));
        drawMesh(cubeMesh, shaderProgram, modelCube2);

        // 5. SZEŚCIAN RUCHOMY - średnio błyszczący
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