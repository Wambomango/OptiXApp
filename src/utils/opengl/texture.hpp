#pragma once


#define GL_CREATE_TEXTURE_2D(texture_handle, datatype, width, height ) \
    glActiveTexture(GL_TEXTURE0); \
    glCreateTextures(GL_TEXTURE_2D, 1, &texture_handle); \
    glTextureStorage2D(texture_handle, 1, datatype, width, height); \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT); \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    