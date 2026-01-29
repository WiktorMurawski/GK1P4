#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <cmath>

#define M_PI 3.14159265358979323846

const char* TITLE = "GK1P4 - Faza 2: Kamery";

// =====================================================================
//   GLOBALNE ZMIENNE DLA KAMER
// =====================================================================
int activeCamera = 0; // 0 = obserwująca, 1 = śledząca, 2 = TPP

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
//   SHADERY (na razie proste, rozbudujemy w Fazie 3)
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
//   GENEROWANIE GEOMETRII
// =====================================================================

// Tworzy sześcian (cube)
Mesh createCube()
{
    // Wierzchołki sześcianu z kolorami (pozycja XYZ, kolor RGB)
    float vertices[] = {
        // Przód (czerwony)
        -0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,   1.0f, 0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,   1.0f, 0.0f, 0.0f,

        // Tył (zielony)
        -0.5f, -0.5f, -0.5f,   0.0f, 1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,   0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,   0.0f, 1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,   0.0f, 1.0f, 0.0f,

        // Góra (niebieski)
        -0.5f,  0.5f, -0.5f,   0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,   0.0f, 0.0f, 1.0f,
         0.5f,  0.5f,  0.5f,   0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,   0.0f, 0.0f, 1.0f,

        // Dół (żółty)
        -0.5f, -0.5f, -0.5f,   1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,   1.0f, 1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,   1.0f, 1.0f, 0.0f,
         0.5f, -0.5f, -0.5f,   1.0f, 1.0f, 0.0f,

        // Prawa (cyjan)
         0.5f, -0.5f, -0.5f,   0.0f, 1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,   0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,   0.0f, 1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,   0.0f, 1.0f, 1.0f,

        // Lewa (magenta)
        -0.5f, -0.5f, -0.5f,   1.0f, 0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,   1.0f, 0.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,   1.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,   1.0f, 0.0f, 1.0f
    };

    unsigned int indices[] = {
        0, 1, 2,  2, 3, 0,      // Przód
        4, 6, 5,  6, 4, 7,      // Tył
        8, 9, 10, 10, 11, 8,    // Góra
        12, 14, 13, 14, 12, 15, // Dół
        16, 17, 18, 18, 19, 16, // Prawa
        20, 22, 21, 22, 20, 23  // Lewa
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Kolor
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return mesh;
}

// Tworzy płaszczyznę (podłoga)
Mesh createPlane()
{
    float vertices[] = {
        // Pozycja              Kolor (szary)
        -10.0f, 0.0f, -10.0f,   0.3f, 0.3f, 0.3f,
         10.0f, 0.0f, -10.0f,   0.3f, 0.3f, 0.3f,
         10.0f, 0.0f,  10.0f,   0.3f, 0.3f, 0.3f,
        -10.0f, 0.0f,  10.0f,   0.3f, 0.3f, 0.3f
    };

    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return mesh;
}

// Tworzy kulę metodą UV sphere
Mesh createSphere(int stacks = 20, int slices = 20)
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

            // Kolor zależny od pozycji (gradient)
            float r = (x + 1.0f) * 0.5f;
            float g = (y + 1.0f) * 0.5f;
            float b = (z + 1.0f) * 0.5f;

            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
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
            indices.push_back(second);
            indices.push_back(first + 1);

            // Drugi trójkąt
            indices.push_back(second);
            indices.push_back(second + 1);
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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

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

    // ==================== Ustawienia OpenGL ====================
    glEnable(GL_DEPTH_TEST);
    // Face culling wyłączony na razie - włączymy w Fazie 3 z poprawnymi normalami
    // glEnable(GL_CULL_FACE);

    // ==================== Pętla główna ====================
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.1f, 0.15f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        float time = (float)glfwGetTime();

        // ==================== OBLICZANIE POZYCJI RUCHOMEGO OBIEKTU ====================
        // (potrzebne dla kamer śledzącej i TPP)
        float radius = 4.0f;
        float speed = 0.5f;
        float movingX = radius * cos(time * speed);
        float movingZ = radius * sin(time * speed);
        float movingY = 1.0f;
        glm::vec3 movingObjPosition(movingX, movingY, movingZ);

        // Kierunek ruchu obiektu (do kamery TPP)
        glm::vec3 movingDirection = glm::normalize(glm::vec3(-sin(time * speed), 0.0f, cos(time * speed)));

        // ==================== MACIERZE ====================

        // MACIERZ VIEW - w zależności od aktywnej kamery
        glm::mat4 view;

        if (activeCamera == 0)
        {
            // KAMERA 0: Obserwująca - stała pozycja, patrzy na środek sceny
            view = glm::lookAt(
                glm::vec3(5.0f, 5.0f, 10.0f),  // Pozycja kamery
                glm::vec3(0.0f, 0.0f, 0.0f),   // Patrzy na środek
                glm::vec3(0.0f, 1.0f, 0.0f)    // Góra
            );
        }
        else if (activeCamera == 1)
        {
            // KAMERA 1: Śledząca - stała pozycja, patrzy na ruchomy obiekt
            view = glm::lookAt(
                glm::vec3(8.0f, 6.0f, 8.0f),   // Stała pozycja (z góry i z boku)
                movingObjPosition,              // Patrzy na ruchomy obiekt
                glm::vec3(0.0f, 1.0f, 0.0f)    // Góra
            );
        }
        else if (activeCamera == 2)
        {
            // KAMERA 2: TPP (Third Person) - podąża za obiektem z tyłu
            float cameraDistance = 3.0f;
            float cameraHeight = 2.0f;

            // Pozycja kamery za obiektem (przeciwnie do kierunku ruchu)
            glm::vec3 cameraPos = movingObjPosition - movingDirection * cameraDistance;
            cameraPos.y += cameraHeight;

            view = glm::lookAt(
                cameraPos,                      // Za obiektem i wyżej
                movingObjPosition,              // Patrzy na obiekt
                glm::vec3(0.0f, 1.0f, 0.0f)    // Góra
            );
        }

        // Rzutowanie perspektywiczne
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f),
            (float)width / (float)height,
            0.1f, 100.0f
        );

        // Przekazanie macierzy widoku i rzutowania (wspólne dla wszystkich obiektów)
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // ==================== RYSOWANIE OBIEKTÓW ====================

        // 1. PODŁOGA (statyczna)
        glm::mat4 modelFloor = glm::mat4(1.0f);
        modelFloor = glm::translate(modelFloor, glm::vec3(0.0f, -2.0f, 0.0f));
        drawMesh(planeMesh, shaderProgram, modelFloor);

        // 2. KULA (statyczna, gładki obiekt)
        glm::mat4 modelSphere = glm::mat4(1.0f);
        modelSphere = glm::translate(modelSphere, glm::vec3(-2.0f, 0.5f, 2.0f));
        modelSphere = glm::scale(modelSphere, glm::vec3(1.0f, 1.0f, 1.0f));
        drawMesh(sphereMesh, shaderProgram, modelSphere);

        // 3. SZEŚCIAN STATYCZNY #1
        glm::mat4 modelCube1 = glm::mat4(1.0f);
        modelCube1 = glm::translate(modelCube1, glm::vec3(2.0f, 1.0f, -2.0f));
        modelCube1 = glm::rotate(modelCube1, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        drawMesh(cubeMesh, shaderProgram, modelCube1);

        // 4. SZEŚCIAN STATYCZNY #2
        glm::mat4 modelCube2 = glm::mat4(1.0f);
        modelCube2 = glm::translate(modelCube2, glm::vec3(3.0f, 0.5f, 3.0f));
        modelCube2 = glm::scale(modelCube2, glm::vec3(0.8f, 1.5f, 0.8f));
        drawMesh(cubeMesh, shaderProgram, modelCube2);

        // 5. SZEŚCIAN RUCHOMY (przesuwanie + obroty)
        glm::mat4 modelMoving = glm::mat4(1.0f);

        // Używamy wcześniej obliczonej pozycji
        modelMoving = glm::translate(modelMoving, movingObjPosition);

        // Obroty
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