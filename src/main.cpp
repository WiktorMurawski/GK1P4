#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>

const char* TITLE = "GK1P4";

// =====================================================================
//   SHADERY 
// =====================================================================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;
out vec3 worldPos;
out vec3 worldNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float time;           
uniform float baseScale;

void main()
{
    // PULSOWANIE
    float pulse = 1.0f + 0.3f * sin(time * 2.0f);
    float scale = baseScale * pulse; 
    vec3 scaledPos = aPos * scale;

    vec4 world = model * vec4(scaledPos, 1.0);
    gl_Position = projection * view * world;

    worldPos = world.xyz;
    worldNormal = normalize(mat3(transpose(inverse(model))) * aPos); 

    ourColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec3 ourColor;
in vec3 worldPos;
in vec3 worldNormal;

uniform vec3 cameraPos;

uniform float time;
uniform float alpha;
uniform float fresnelPower;

void main()
{
    // ZMIANA KOLORU W CZASIE
    vec3 color = ourColor + 0.5 * sin(0.5 * time + vec3(0.0, 2.0, 4.0)); 

    vec3 view = normalize(cameraPos - worldPos);
    vec3 norm = normalize(worldNormal);

    float NdotV = min(max(dot(norm, view), 0.0), 1.0);
    float fresnel = 1.0 - NdotV;

    fresnel = pow(fresnel, fresnelPower);

    color += vec3(0.5, 0.5, 0.5) * fresnel;  
    float finalAlpha = alpha + 0.5 * fresnel;           

    FragColor = vec4(color, finalAlpha);
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
    unsigned int vs = compileShader(vertexShaderSource,   GL_VERTEX_SHADER);
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
//   DANE GEOMETRII – regularny czworościan
// =====================================================================
float vertices[] = {
    // pozycja              // kolor
     1.0f,  1.0f,  1.0f,    1.0f, 0.0f, 0.0f,   // 0 czerwony
    -1.0f, -1.0f,  1.0f,    0.0f, 1.0f, 0.0f,   // 1 zielony
    -1.0f,  1.0f, -1.0f,    0.0f, 0.0f, 1.0f,   // 2 niebieski
     1.0f, -1.0f, -1.0f,    1.0f, 1.0f, 0.0f    // 3 żółty
};

unsigned int indices[] = {
    0, 1, 2,
    0, 3, 1,
    0, 2, 3,
    1, 3, 2
};

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

    if (!gladLoadGL()) {
        std::cerr << "Nie udało się zainicjalizować GLAD / OpenGL" << std::endl;
        return -1;
    }

    // Wypisanie wersji OpenGL
    int major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "Załadowano OpenGL " << major << "." << minor << std::endl;

    // WŁĄCZENIE BLENDINGU
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ==================== Kompilacja i linkowanie shaderów ====================
    unsigned int shaderProgram = createShaderProgram();

    // ==================== VAO / VBO / EBO ====================
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // aPos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // aColor
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // ==================== Ustawienia OpenGL ====================
    glEnable(GL_DEPTH_TEST);

    // ==================== Pętla główna ====================
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.08f, 0.12f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        float time = (float)glfwGetTime();
        
        // Przekazanie czasu do shaderów
        glUniform1f(glGetUniformLocation(shaderProgram, "time"), time);

        // Macierze
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, time * 0.4f, glm::vec3(1.0f, 0.0f, 0.0f));

        glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 5.0f);

        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 5.0f),   // kamera
            glm::vec3(0.0f, 0.0f, 0.0f),   // patrzy na środek
            glm::vec3(0.0f, 1.0f, 0.0f)    // góra
        );

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glm::mat4 projection = glm::perspective(
            glm::radians(55.0f),
            (float)width / (float)height,
            0.1f, 100.0f
        );

        // Przekazanie macierzy do shadera
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "cameraPos"), 1, glm::value_ptr(cameraPosition));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
       
        // ============================
        // 1. Wewnętrzny czworościan
        // ============================
        glUniform1f(glGetUniformLocation(shaderProgram, "baseScale"), 1.0f);
        glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 1.0f);
        glUniform1f(glGetUniformLocation(shaderProgram, "fresnelPower"), 2.0f);

        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CW);
    
        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);
    
        // ============================
        // 2. Zewnętrzna skorupa 
        // ============================
        glUniform1f(glGetUniformLocation(shaderProgram, "baseScale"), 1.20f); 
        glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), 0.1f); 
        glUniform1f(glGetUniformLocation(shaderProgram, "fresnelPower"), 3.0f);

        glEnable(GL_CULL_FACE);
        // glCullFace(GL_FRONT);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);
        // glFrontFace(GL_CW);

        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ==================== Sprzątanie ====================
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
