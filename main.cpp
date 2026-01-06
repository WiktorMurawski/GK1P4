#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>

// =====================================================================
//   SHADERY (można też trzymać w osobnych plikach .vert / .frag)
// =====================================================================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
in vec3 ourColor;

void main()
{
    FragColor = vec4(ourColor, 1.0);
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
    // pozycja             // kolor
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

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1200, 900, "Czworościan OpenGL", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Nie udało się utworzyć okna GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0)
    {
        std::cerr << "Nie udało się zainicjalizować GLAD / OpenGL" << std::endl;
        return -1;
    }

    std::cout << "Załadowano OpenGL " 
              << GLAD_VERSION_MAJOR(version) << "." 
              << GLAD_VERSION_MINOR(version) << std::endl;


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

    glBindVertexArray(0);  // odpinamy VAO

    // ==================== Ustawienia OpenGL ====================
    glEnable(GL_DEPTH_TEST);

    // ==================== Pętla główna ====================
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.08f, 0.12f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // Macierze
        glm::mat4 model = glm::mat4(1.0f);
        float time = (float)glfwGetTime();
        model = glm::rotate(model, time * 0.8f, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, time * 0.4f, glm::vec3(1.0f, 0.0f, 0.0f));

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
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"),      1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"),       1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Rysowanie
        glBindVertexArray(VAO);
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
