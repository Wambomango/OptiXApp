#include "scene.hpp"
#include <torch/script.h>


Scene::Scene(std::string path)
{
    this->path = path;

    std::filesystem::path fullpath(this->path);
    if (fullpath.extension() == ".obj") 
    {
        LoadFromObj(vertices, indices);

    } 
    else if (fullpath.extension() == ".pt") 
    {
        LoadFromPt(vertices, indices);
    } 
    else 
    {
        throw std::runtime_error("Unsupported file format: " + fullpath.extension().string());
    }
}

void Scene::GetMesh(torch::Tensor &vertices, torch::Tensor &indices)
{
    vertices = this->vertices;
    indices = this->indices;
}

void Scene::LoadFromObj(torch::Tensor &vertices, torch::Tensor &indices)
{
    std::filesystem::path fullpath(path);

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = fullpath.remove_filename();
    reader_config.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, reader_config)) 
    {
        if (!reader.Error().empty()) 
        {
            SPDLOG_ERROR("TinyObjReader {}", reader.Error());
        }

        exit(1);
    }

    if (!reader.Warning().empty()) 
    {
        SPDLOG_WARN("TinyObjReader: {}", reader.Warning());
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    int n_vertices = attrib.vertices.size() / 3;
    int n_indices = 0;
    for (const auto& shape : shapes)
    {
        n_indices += shape.mesh.indices.size();
    }
    int n_faces = n_indices / 3;

    vertices = torch::empty({n_vertices, 3}, torch::kFloat32);
    memcpy(vertices.data_ptr(), attrib.vertices.data(), n_vertices * 3 * sizeof(float));
    indices = torch::empty({n_faces, 3}, torch::kUInt32);
    uint32_t *index_data = (uint32_t *)indices.data_ptr();
    int index_offset = 0;
    for(const auto &shape : shapes)
    {
        for(int i = 0; i < shape.mesh.indices.size() / 3; i++)
        {
            index_data[index_offset++] = shape.mesh.indices[3 * i + 0].vertex_index;
            index_data[index_offset++] = shape.mesh.indices[3 * i + 1].vertex_index;
            index_data[index_offset++] = shape.mesh.indices[3 * i + 2].vertex_index;
        }
    }
}

void Scene::LoadFromPt(torch::Tensor &vertices, torch::Tensor &indices)
{
    torch::jit::script::Module container = torch::jit::load(path);
    vertices = container.attr("vertices").toTensor();
    indices = container.attr("indices").toTensor();

}
