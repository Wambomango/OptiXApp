#pragma once

#include <tiny_obj_loader.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <glm/glm.hpp>

class Scene
{
    public:
        struct SceneVertex
        {
            glm::vec3 position;
            glm::vec3 normal;
        };
    
        Scene(const std::string &path)
        {
            std::filesystem::path fullpath(path);
     
            tinyobj::ObjReaderConfig reader_config;
            reader_config.mtl_search_path = fullpath.remove_filename(); // Path to material files


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
        }

        const tinyobj::attrib_t &GetAttrib()
        {
            return reader.GetAttrib();
        }
        
        const std::vector<tinyobj::shape_t> &GetShapes()
        { 
            return reader.GetShapes();
        }

        const std::vector<tinyobj::material_t> &GetMaterials()
        { 
            return reader.GetMaterials(); 
        }

    
    private:
        tinyobj::ObjReader reader;

};