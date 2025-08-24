#pragma once

#include "vec_math.h"

struct complex
{
    float real;
    float imag;
};

struct complex2
{
    complex x;
    complex y;
};

struct complex3
{
    complex x;
    complex y;
    complex z;
};

struct complex4
{
    complex x;
    complex y;
    complex z;
    complex w;
};


__host__ __device__ complex2 make_complex2(const complex &x, const complex &y);
__host__ __device__ complex3 make_complex3(const complex &x, const complex &y, const complex &z);
__host__ __device__ complex4 make_complex4(const complex &x, const complex &y, const complex &z, const complex &w);



/* complex functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
__forceinline__ __host__ __device__ complex make_complex(float real, float imag)
{
  return complex{real, imag};
}
__forceinline__ __host__ __device__ complex make_complex(const complex c)
{
  return make_complex(c.real, c.imag);
}
__forceinline__ __host__ __device__ complex make_complex(const int2& a)
{
  return make_complex(float(a.x), float(a.y));
}
__forceinline__ __host__ __device__ complex make_complex(const uint2& a)
{
  return make_complex(float(a.x), float(a.y));
}
/** @} */


/** negate */
__forceinline__ __host__ __device__ complex operator-(const complex& a)
{
  return make_complex(-a.real, -a.imag);
}

/** add 
* @{
*/
__forceinline__ __host__ __device__ complex operator+(const complex& a, const complex& b)
{
  return make_complex(a.real + b.real, a.imag + b.imag);
}
__forceinline__ __host__ __device__ complex operator+(const complex& a, const float b)
{
  return make_complex(a.real + b, a.imag + b);
}
__forceinline__ __host__ __device__ complex operator+(const float a, const complex& b)
{
  return make_complex(a + b.real, b.imag);
}
__forceinline__ __host__ __device__ complex2 operator+(const complex& a, const float2 b)
{
  return make_complex2(a + b.x, a + b.y);
}
__forceinline__ __host__ __device__ complex2 operator+(const float2 a, const complex& b)
{
  return make_complex2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ complex3 operator+(const complex& a, const float3 b)
{
  return make_complex3(a + b.x, a + b.y, a + b.z);
}
__forceinline__ __host__ __device__ complex3 operator+(const float3 a, const complex& b)
{
  return make_complex3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ complex4 operator+(const complex& a, const float4 b)
{
  return make_complex4(a + b.x, a + b.y, a + b.z, a + b.w);
}
__forceinline__ __host__ __device__ complex4 operator+(const float4 a, const complex& b)
{
  return make_complex4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ void operator+=(complex& a, const complex& b)
{
  a.real += b.real; a.imag += b.imag;
}
__forceinline__ __host__ __device__ void operator+=(complex& a, const float& b)
{
  a.real += b;
}



/** @} */

/** subtract 
* @{
*/
__forceinline__ __host__ __device__ complex operator-(const complex& a, const complex& b)
{
  return make_complex(a.real - b.real, a.imag - b.imag);
}
__forceinline__ __host__ __device__ complex operator-(const complex& a, const float b)
{
  return make_complex(a.real - b, a.imag);
}
__forceinline__ __host__ __device__ complex operator-(const float a, const complex& b)
{
  return make_complex(a - b.real, - b.imag);
}
__forceinline__ __host__ __device__ complex2 operator-(const complex& a, const float2 b)
{
  return make_complex2(a - b.x, a - b.y);
}
__forceinline__ __host__ __device__ complex2 operator-(const float2 a, const complex& b)
{
  return make_complex2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ complex3 operator-(const complex& a, const float3 b)
{
  return make_complex3(a - b.x, a - b.y, a - b.z);
}
__forceinline__ __host__ __device__ complex3 operator-(const float3 a, const complex& b)
{
  return make_complex3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ complex4 operator-(const complex& a, const float4 b)
{
  return make_complex4(a - b.x, a - b.y, a - b.z, a - b.w);
}
__forceinline__ __host__ __device__ complex4 operator-(const float4 a, const complex& b)
{
  return make_complex4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ void operator-=(complex& a, const complex& b)
{
  a.real -= b.real; a.imag -= b.imag;
}
__forceinline__ __host__ __device__ void operator-=(complex& a, const float& b)
{
  a.real -= b;
}

/** @} */


/** multiply 
* @{
*/
__forceinline__ __host__ __device__ complex operator*(const complex& a, const complex& b)
{
  return make_complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}
__forceinline__ __host__ __device__ complex operator*(const complex& a, const float s)
{
  return make_complex(a.real * s, a.imag * s);
}
__forceinline__ __host__ __device__ complex operator*(const float s, const complex& a)
{
  return make_complex(a.real * s, a.imag * s);
}
__forceinline__ __host__ __device__ complex2 operator*(const complex& a, const float2 b)
{
  return make_complex2(a * b.x, a * b.y);
}
__forceinline__ __host__ __device__ complex2 operator*(const float2 a, const complex& b)
{
  return make_complex2(a.x * b, a.y * b);
}
__forceinline__ __host__ __device__ complex3 operator*(const complex& a, const float3 b)
{
  return make_complex3(a * b.x, a * b.y, a * b.z);
}
__forceinline__ __host__ __device__ complex3 operator*(const float3 a, const complex& b)
{
  return make_complex3(a.x * b, a.y * b, a.z * b);
}
__forceinline__ __host__ __device__ complex4 operator*(const complex& a, const float4 b)
{
  return make_complex4(a * b.x, a * b.y, a * b.z, a * b.w);
}
__forceinline__ __host__ __device__ complex4 operator*(const float4 a, const complex& b)
{
  return make_complex4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__forceinline__ __host__ __device__ void operator*=(complex& a, const complex& s)
{
    float real = a.real;
    a.real = a.real * s.real - a.imag * s.imag;
    a.imag = real * s.imag + a.imag * s.real;
}
__forceinline__ __host__ __device__ void operator*=(complex& a, const float s)
{
  a.real *= s; a.imag *= s;
}

/** @} */

/** divide 
* @{
*/
__forceinline__ __host__ __device__ complex operator/(const complex& a, const complex& b)
{
  float denom = b.real * b.real + b.imag * b.imag;
  return make_complex((a.real * b.real + a.imag * b.imag) / denom,
                      (a.imag * b.real - a.real * b.imag) / denom);
}
__forceinline__ __host__ __device__ complex operator/(const complex& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__forceinline__ __host__ __device__ complex operator/(const float s, const complex& a)
{
  float denom = a.real * a.real + a.imag * a.imag;
  return make_complex(s * a.real / denom, -s * a.imag / denom);
}
__forceinline__ __host__ __device__ complex2 operator/(const complex& a, const float2 b)
{
  return make_complex2(a / b.x, a / b.y);
}
__forceinline__ __host__ __device__ complex2 operator/(const float2 a, const complex& b)
{
  return make_complex2(a.x / b, a.y / b);
}
__forceinline__ __host__ __device__ complex3 operator/(const complex& a, const float3 b)
{
  return make_complex3(a / b.x, a / b.y, a / b.z);
}
__forceinline__ __host__ __device__ complex3 operator/(const float3 a, const complex& b)
{
  return make_complex3(a.x / b, a.y / b, a.z / b);
}
__forceinline__ __host__ __device__ complex4 operator/(const complex& a, const float4 b)
{
  return make_complex4(a / b.x, a / b.y, a / b.z, a / b.w);
}
__forceinline__ __host__ __device__ complex4 operator/(const float4 a, const complex& b)
{
  return make_complex4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__forceinline__ __host__ __device__ void operator/=(complex& a, const complex& s)
{
    complex inv = 1.0f / s;
    a *= inv;
}
__forceinline__ __host__ __device__ void operator/=(complex& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}

/** @} */


/** magnitude */
__forceinline__ __host__ __device__ float magnitude(const complex& a)
{
  return sqrtf(a.real * a.real + a.imag * a.imag);
}

/** phase */
__forceinline__ __host__ __device__ float phase(const complex& a)
{
  return atan2f(a.imag, a.real);
}

/** conjugate */
__forceinline__ __host__ __device__ complex conj(const complex& a)
{
  return make_complex(a.real, -a.imag);
}

/** normalize */
__forceinline__ __host__ __device__ complex normalize(const complex& a)
{
  float invLen = 1.0f / sqrtf(a.real * a.real + a.imag * a.imag);
  return make_complex(a.real * invLen, a.imag * invLen);
}

/** exp */
__forceinline__ __host__ __device__ complex expf(const complex& a)
{
  return make_complex(expf(a.real) * cosf(a.imag), expf(a.real) * sinf(a.imag));
}






/* complex2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/

__forceinline__ __host__ __device__ complex2 make_complex2(const complex &a, const complex &b)
{
  return complex2{a, b};
}
__forceinline__ __host__ __device__ complex2 make_complex2(const float2 &a)
{
  return complex2{make_complex(a.x, 0), make_complex(a.y, 0)};
}
__forceinline__ __host__ __device__ complex2 make_complex2(const complex s)
{
  return make_complex2(s, s);
}
__forceinline__ __host__ __device__ complex2 make_complex2(float x, float y)
{
  return make_complex2(make_complex(x, 0), make_complex(y, 0));
}
__forceinline__ __host__ __device__ complex2 make_complex2(const float& c)
{
  return make_complex2(make_complex(c, 0), make_complex(c, 0));
}


/** @} */

/** negate */
__forceinline__ __host__ __device__ complex2 operator-(const complex2& a)
{
  return make_complex2(-a.x, -a.y);
}

/** add 
* @{
*/
__forceinline__ __host__ __device__ complex2 operator+(const complex2& a, const complex2& b)
{
  return make_complex2(a.x + b.x, a.y + b.y);
}
__forceinline__ __host__ __device__ complex2 operator+(const complex2& a, const complex b)
{
  return make_complex2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ complex2 operator+(const complex& a, const complex2& b)
{
  return make_complex2(a + b.x, a + b.y);
}
__forceinline__ __host__ __device__ complex2 operator+(const complex2& a, const float b)
{
  return make_complex2(a.x + b, a.y + b);
}
__forceinline__ __host__ __device__ complex2 operator+(const float a, const complex2& b)
{
  return make_complex2(a + b.x, a + b.y);
}
__forceinline__ __host__ __device__ void operator+=(complex2& a, const complex2& b)
{
  a.x += b.x; a.y += b.y;
}
__forceinline__ __host__ __device__ void operator+=(complex2& a, const complex& b)
{
  a.x += b; a.y += b;
}
__forceinline__ __host__ __device__ void operator+=(complex2& a, const float& b)
{
  a.x += b; a.y += b;
}


/** @} */

/** subtract 
* @{
*/
__forceinline__ __host__ __device__ complex2 operator-(const complex2& a, const complex2& b)
{
  return make_complex2(a.x - b.x, a.y - b.y);
}
__forceinline__ __host__ __device__ complex2 operator-(const complex2& a, const complex b)
{
  return make_complex2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ complex2 operator-(const complex a, const complex2& b)
{
  return make_complex2(a - b.x, a - b.y);
}
__forceinline__ __host__ __device__ complex2 operator-(const complex2& a, const float b)
{
  return make_complex2(a.x - b, a.y - b);
}
__forceinline__ __host__ __device__ complex2 operator-(const float a, const complex2& b)
{
  return make_complex2(a - b.x, a - b.y);
}
__forceinline__ __host__ __device__ void operator-=(complex2& a, const complex2& b)
{
  a.x -= b.x; a.y -= b.y;
}
__forceinline__ __host__ __device__ void operator-=(complex2& a, const complex& b)
{
  a.x -= b; a.y -= b;
}
__forceinline__ __host__ __device__ void operator-=(complex2& a, const float& b)
{
  a.x -= b; a.y -= b;
}

/** @} */

/** multiply 
* @{
*/
__forceinline__ __host__ __device__ complex2 operator*(const complex2& a, const complex2& b)
{
  return make_complex2(a.x * b.x, a.y * b.y);
}
__forceinline__ __host__ __device__ complex2 operator*(const complex2& a, const complex s)
{
  return make_complex2(a.x * s, a.y * s);
}
__forceinline__ __host__ __device__ complex2 operator*(const complex s, const complex2& a)
{
  return make_complex2(a.x * s, a.y * s);
}
__forceinline__ __host__ __device__ complex2 operator*(const complex2& a, const float s)
{
  return make_complex2(a.x * s, a.y * s);
}
__forceinline__ __host__ __device__ complex2 operator*(const float s, const complex2& a)
{
  return make_complex2(a.x * s, a.y * s);
}
__forceinline__ __host__ __device__ void operator*=(complex2& a, const complex2& s)
{
  a.x *= s.x; a.y *= s.y;
}
__forceinline__ __host__ __device__ void operator*=(complex2& a, const complex &s)
{
  a.x *= s; a.y *= s;
}
__forceinline__ __host__ __device__ void operator*=(complex2& a, const float s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** divide 
* @{
*/
__forceinline__ __host__ __device__ complex2 operator/(const complex2& a, const complex2& b)
{
  return make_complex2(a.x / b.x, a.y / b.y);
}
__forceinline__ __host__ __device__ complex2 operator/(const complex2& a, const complex &s)
{
  complex inv = 1.0f / s;
  return make_complex2(a.x * inv, a.y * inv);
}
__forceinline__ __host__ __device__ complex2 operator/(const complex s, const complex2& a)
{
  return make_complex2(s / a.x, s / a.y);
}
__forceinline__ __host__ __device__ complex2 operator/(const complex2& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__forceinline__ __host__ __device__ complex2 operator/(const float s, const complex2& a)
{
  return make_complex2( s / a.x, s / a.y );
}
__forceinline__ __host__ __device__ void operator/=(complex2& a, const complex2 &s)
{
    a.x /= s.x; a.y /= s.y;
}
__forceinline__ __host__ __device__ void operator/=(complex2& a, const complex &s)
{
  complex inv = 1.0f / s;
  a *= inv;
}
__forceinline__ __host__ __device__ void operator/=(complex2& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** magnitude */
__forceinline__ __host__ __device__ float2 magnitude(const complex2& a)
{
  return make_float2(magnitude(a.x), magnitude(a.y));
}

/** phase */
__forceinline__ __host__ __device__ float2 phase(const complex2& a)
{
  return make_float2(phase(a.x), phase(a.y));
}

/** conjugate */
__forceinline__ __host__ __device__ complex2 conj(const complex2& a)
{
  return make_complex2(conj(a.x), conj(a.y));
}

/** exp */
__forceinline__ __host__ __device__ complex2 expf(const complex2& a)
{
  return make_complex2(expf(a.x), expf(a.y));
}

/** dot */
__forceinline__ __host__ __device__ complex dot(const complex2& a, const complex2& b)
{
  return a.x * conj(b.x) + a.y * conj(b.y);
}

__forceinline__ __host__ __device__ complex elsum(const complex2& a)
{
  return a.x + a.y;
}

__forceinline__ __host__ __device__ float2 real(const complex2& a)
{
  return make_float2(a.x.real, a.y.real);
}

__forceinline__ __host__ __device__ float2 imag(const complex2& a)
{
  return make_float2(a.x.imag, a.y.imag);
}





/* complex3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/

__forceinline__ __host__ __device__ complex3 make_complex3(const complex &a, const complex &b, const complex &c)
{
  return complex3{a, b, c};
}
__forceinline__ __host__ __device__ complex3 make_complex3(const float3 &a)
{
  return complex3{make_complex(a.x, 0), make_complex(a.y, 0), make_complex(a.z, 0)};
}
__forceinline__ __host__ __device__ complex3 make_complex3(const complex s)
{
  return make_complex3(s, s, s);
}
__forceinline__ __host__ __device__ complex3 make_complex3(float x, float y, float z)
{
  return complex3{make_complex(x, 0), make_complex(y, 0), make_complex(z, 0)};
}
__forceinline__ __host__ __device__ complex3 make_complex3(const float& c)
{
  return make_complex3(make_complex(c, 0), make_complex(c, 0), make_complex(c, 0));
}


/** @} */

/** negate */
__forceinline__ __host__ __device__ complex3 operator-(const complex3& a)
{
  return make_complex3(-a.x, -a.y, -a.z);
}

/** add 
* @{
*/
__forceinline__ __host__ __device__ complex3 operator+(const complex3& a, const complex3& b)
{
  return make_complex3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __host__ __device__ complex3 operator+(const complex3& a, const complex& b)
{
  return make_complex3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ complex3 operator+(const complex& a, const complex3& b)
{
  return make_complex3(a + b.x, a + b.y, a + b.z);
}
__forceinline__ __host__ __device__ complex3 operator+(const complex3& a, const float b)
{
  return make_complex3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __host__ __device__ complex3 operator+(const float a, const complex3& b)
{
  return make_complex3(a + b.x, a + b.y, a + b.z);
}
__forceinline__ __host__ __device__ void operator+=(complex3& a, const complex3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
__forceinline__ __host__ __device__ void operator+=(complex3& a, const complex& b)
{
  a.x += b; a.y += b; a.z += b;
}
__forceinline__ __host__ __device__ void operator+=(complex3& a, const float& b)
{
  a.x += b; a.y += b; a.z += b;
}


/** @} */

/** subtract 
* @{
*/
__forceinline__ __host__ __device__ complex3 operator-(const complex3& a, const complex3& b)
{
  return make_complex3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __host__ __device__ complex3 operator-(const complex3& a, const complex& b)
{
  return make_complex3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ complex3 operator-(const complex a, const complex3& b)
{
  return make_complex3(a - b.x, a - b.y, a - b.z);
}
__forceinline__ __host__ __device__ complex3 operator-(const complex3& a, const float b)
{
  return make_complex3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __host__ __device__ complex3 operator-(const float a, const complex3& b)
{
  return make_complex3(a - b.x, a - b.y, a - b.z);
}
__forceinline__ __host__ __device__ void operator-=(complex3& a, const complex3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
__forceinline__ __host__ __device__ void operator-=(complex3& a, const complex& b)
{
  a.x -= b; a.y -= b; a.z -= b;
}
__forceinline__ __host__ __device__ void operator-=(complex3& a, const float& b)
{
  a.x -= b; a.y -= b; a.z -= b;
}

/** @} */

/** multiply 
* @{
*/
__forceinline__ __host__ __device__ complex3 operator*(const complex3& a, const complex3& b)
{
  return make_complex3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __host__ __device__ complex3 operator*(const complex3& a, const complex s)
{
  return make_complex3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __host__ __device__ complex3 operator*(const complex s, const complex3& a)
{
  return make_complex3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __host__ __device__ complex3 operator*(const complex3& a, const float s)
{
  return make_complex3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __host__ __device__ complex3 operator*(const float s, const complex3& a)
{
  return make_complex3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __host__ __device__ void operator*=(complex3& a, const complex3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
__forceinline__ __host__ __device__ void operator*=(complex3& a, const complex& s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
__forceinline__ __host__ __device__ void operator*=(complex3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
__forceinline__ __host__ __device__ complex3 operator/(const complex3& a, const complex3& b)
{
  return make_complex3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__forceinline__ __host__ __device__ complex3 operator/(const complex3& a, const complex& s)
{
  complex inv = 1.0f / s;
  return make_complex3(a.x * inv, a.y * inv, a.z * inv);
}
__forceinline__ __host__ __device__ complex3 operator/(const complex& s, const complex3& a)
{
  return make_complex3(s / a.x, s / a.y, s / a.z);
}
__forceinline__ __host__ __device__ complex3 operator/(const complex3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__forceinline__ __host__ __device__ complex3 operator/(const float s, const complex3& a)
{
  return make_complex3(s / a.x, s / a.y, s / a.z);
}
__forceinline__ __host__ __device__ void operator/=(complex3& a, const complex3& s)
{
    a.x /= s.x; a.y /= s.y; a.z /= s.z;
}
__forceinline__ __host__ __device__ void operator/=(complex3& a, const complex& s)
{
  complex inv = 1.0f / s;
  a *= inv;
}
__forceinline__ __host__ __device__ void operator/=(complex3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** magnitude */
__forceinline__ __host__ __device__ float3 magnitude(const complex3& a)
{
  return make_float3(magnitude(a.x), magnitude(a.y), magnitude(a.z));
}

/** phase */
__forceinline__ __host__ __device__ float3 phase(const complex3& a)
{
  return make_float3(phase(a.x), phase(a.y), phase(a.z));
}

/** conjugate */
__forceinline__ __host__ __device__ complex3 conj(const complex3& a)
{
  return make_complex3(conj(a.x), conj(a.y), conj(a.z));
}

/** exp */
__forceinline__ __host__ __device__ complex3 expf(const complex3& a)
{
  return make_complex3(expf(a.x), expf(a.y), expf(a.z));
}

/** dot */
__forceinline__ __host__ __device__ complex dot(const complex3& a, const complex3& b)
{
  return a.x * conj(b.x) + a.y * conj(b.y) + a.z * conj(b.z);
}

__forceinline__ __host__ __device__ complex dot(const complex3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __host__ __device__ complex dot(const float3& a, const complex3& b)
{
  return a.x * conj(b.x) + a.y * conj(b.y) + a.z * conj(b.z);
}



/** cross */
__forceinline__ __host__ __device__ complex3 cross(const complex3& a, const complex3& b)
{
  return make_complex3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__forceinline__ __host__ __device__ complex3 cross(const float3& a, const complex3& b)
{
  return make_complex3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__forceinline__ __host__ __device__ complex3 cross(const complex3& a, const float3& b)
{
  return make_complex3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  );
}

__forceinline__ __host__ __device__ complex elsum(const complex3& a)
{
  return a.x + a.y + a.z;
}

__forceinline__ __host__ __device__ float3 real(const complex3& a)
{
  return make_float3(a.x.real, a.y.real, a.z.real);
}

__forceinline__ __host__ __device__ float3 imag(const complex3& a)
{
  return make_float3(a.x.imag, a.y.imag, a.z.imag);
}





/* complex4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/

__forceinline__ __host__ __device__ complex4 make_complex4(const complex &a, const complex &b, const complex &c, const complex &d)
{
  return complex4{a, b, c, d};
}
__forceinline__ __host__ __device__ complex4 make_complex4(const float4 &a)
{
  return complex4{make_complex(a.x, 0), make_complex(a.y, 0), make_complex(a.z, 0), make_complex(a.w, 0)};
}
__forceinline__ __host__ __device__ complex4 make_complex4(const complex s)
{
  return make_complex4(s, s, s, s);
}
__forceinline__ __host__ __device__ complex4 make_complex4(float x, float y, float z, float w)
{
  return complex4{make_complex(x, 0), make_complex(y, 0), make_complex(z, 0), make_complex(w, 0)};
}
__forceinline__ __host__ __device__ complex4 make_complex4(const float& c)
{
  return make_complex4(make_complex(c, 0), make_complex(c, 0), make_complex(c, 0), make_complex(c, 0));
}


/** @} */

/** negate */
__forceinline__ __host__ __device__ complex4 operator-(const complex4& a)
{
  return make_complex4(-a.x, -a.y, -a.z, -a.w);
}

/** add 
* @{
*/
__forceinline__ __host__ __device__ complex4 operator+(const complex4& a, const complex4& b)
{
  return make_complex4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__forceinline__ __host__ __device__ complex4 operator+(const complex4& a, const complex& b)
{
  return make_complex4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ complex4 operator+(const complex& a, const complex4& b)
{
  return make_complex4(a + b.x, a + b.y, a + b.z, a + b.w);
}
__forceinline__ __host__ __device__ complex4 operator+(const complex4& a, const float b)
{
  return make_complex4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__forceinline__ __host__ __device__ complex4 operator+(const float a, const complex4& b)
{
  return make_complex4(a + b.x, a + b.y, a + b.z, a + b.w);
}
__forceinline__ __host__ __device__ void operator+=(complex4& a, const complex4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
__forceinline__ __host__ __device__ void operator+=(complex4& a, const complex& b)
{
  a.x += b; a.y += b; a.z += b; a.w += b;
}
__forceinline__ __host__ __device__ void operator+=(complex4& a, const float& b)
{
  a.x += b; a.y += b; a.z += b; a.w += b;
}


/** @} */

/** subtract 
* @{
*/
__forceinline__ __host__ __device__ complex4 operator-(const complex4& a, const complex4& b)
{
  return make_complex4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__forceinline__ __host__ __device__ complex4 operator-(const complex4& a, const complex& b)
{
  return make_complex4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ complex4 operator-(const complex a, const complex4& b)
{
  return make_complex4(a - b.x, a - b.y, a - b.z, a - b.w);
}
__forceinline__ __host__ __device__ complex4 operator-(const complex4& a, const float b)
{
  return make_complex4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__forceinline__ __host__ __device__ complex4 operator-(const float a, const complex4& b)
{
  return make_complex4(a - b.x, a - b.y, a - b.z, a - b.w);
}
__forceinline__ __host__ __device__ void operator-=(complex4& a, const complex4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
__forceinline__ __host__ __device__ void operator-=(complex4& a, const complex& b)
{
  a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}
__forceinline__ __host__ __device__ void operator-=(complex4& a, const float& b)
{
  a.x -= b; a.y -= b; a.z -= b; a.w -= b;
}

/** @} */

/** multiply 
* @{
*/
__forceinline__ __host__ __device__ complex4 operator*(const complex4& a, const complex4& b)
{
  return make_complex4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__forceinline__ __host__ __device__ complex4 operator*(const complex4& a, const complex s)
{
  return make_complex4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__forceinline__ __host__ __device__ complex4 operator*(const complex s, const complex4& a)
{
  return make_complex4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__forceinline__ __host__ __device__ complex4 operator*(const complex4& a, const float s)
{
  return make_complex4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__forceinline__ __host__ __device__ complex4 operator*(const float s, const complex4& a)
{
  return make_complex4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__forceinline__ __host__ __device__ void operator*=(complex4& a, const complex4& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
__forceinline__ __host__ __device__ void operator*=(complex4& a, const complex& s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
__forceinline__ __host__ __device__ void operator*=(complex4& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
__forceinline__ __host__ __device__ complex4 operator/(const complex4& a, const complex4& b)
{
  return make_complex4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__forceinline__ __host__ __device__ complex4 operator/(const complex4& a, const complex& s)
{
  complex inv = 1.0f / s;
  return make_complex4(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}
__forceinline__ __host__ __device__ complex4 operator/(const complex& s, const complex4& a)
{
  return make_complex4(s / a.x, s / a.y, s / a.z, s / a.w);
}
__forceinline__ __host__ __device__ complex4 operator/(const complex4& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__forceinline__ __host__ __device__ complex4 operator/(const float s, const complex4& a)
{
  return make_complex4(s / a.x, s / a.y, s / a.z, s / a.w);
}
__forceinline__ __host__ __device__ void operator/=(complex4& a, const complex4& s)
{
    a.x /= s.x; a.y /= s.y; a.z /= s.z; a.w /= s.w;
}
__forceinline__ __host__ __device__ void operator/=(complex4& a, const complex& s)
{
  complex inv = 1.0f / s;
  a *= inv;
}
__forceinline__ __host__ __device__ void operator/=(complex4& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** magnitude */
__forceinline__ __host__ __device__ float4 magnitude(const complex4& a)
{
  return make_float4(magnitude(a.x), magnitude(a.y), magnitude(a.z), magnitude(a.w));
}

/** phase */
__forceinline__ __host__ __device__ float4 phase(const complex4& a)
{
  return make_float4(phase(a.x), phase(a.y), phase(a.z), phase(a.w));
}

/** conjugate */
__forceinline__ __host__ __device__ complex4 conj(const complex4& a)
{
  return make_complex4(conj(a.x), conj(a.y), conj(a.z), conj(a.w));
}

/** exp */
__forceinline__ __host__ __device__ complex4 expf(const complex4& a)
{
  return make_complex4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
}

/** dot */
__forceinline__ __host__ __device__ complex dot(const complex4& a, const complex4& b)
{
  return a.x * conj(b.x) + a.y * conj(b.y) + a.z * conj(b.z) + a.w * conj(b.w);
}

__forceinline__ __host__ __device__ complex elsum(const complex4& a)
{
  return a.x + a.y + a.z + a.w;
}

__forceinline__ __host__ __device__ float4 real(const complex4& a)
{
  return make_float4(a.x.real, a.y.real, a.z.real, a.w.real);
}

__forceinline__ __host__ __device__ float4 imag(const complex4& a)
{
  return make_float4(a.x.imag, a.y.imag, a.z.imag, a.w.imag);
}