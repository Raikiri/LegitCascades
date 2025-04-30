[declaration: "config"]
{{
  int i;
  #define POS_FIRST_LAYOUT 1
}}

[declaration: "pi"]
{{
  const float pi = 3.141592f;
}}

[blendmode: alphablend]
void OverlayTexShader(
  sampler2D tex,
  out vec4 color)
{{
  uvec2 pixel_idx = uvec2(gl_FragCoord.xy);
  uvec2 tex_size = uvec2(textureSize(tex, 0));
  if(pixel_idx.x < tex_size.x && pixel_idx.y < tex_size.y)
  {
    color = vec4(texelFetch(tex, ivec2(pixel_idx), 0).rgb, 1.0);
  }else
  {
    color = vec4(0.0);
  }
}}

[include: "pcg", "complex", "bessel"]
[declaration: "scene"]
{{
  float wavelength = 10.0f;
  Complex GetSceneField(uvec2 scene_size, vec2 pos)
  {
    vec2 center_pos = vec2(scene_size) / 2.0f;
    float l = length(pos - center_pos + vec2(0.0f, 100.0f));
    float phase = l / wavelength * 2.0f * pi;
    return Complex(bessj0(phase), bessy0(phase));
  }
}}

[declaration: "line_grid"]
{{
  vec2 GetGridPointPos(uvec2 scene_size, uvec2 field_size, vec2 point_idx)
  {
    return (point_idx + vec2(0.5f)) / vec2(field_size) * vec2(scene_size);
  }
  vec2 GetGridPointIdx(uvec2 scene_size, uvec2 field_size, vec2 point_pos)
  {
    return point_pos / vec2(scene_size) * vec2(field_size) - vec2(0.5f);
  }
}}

void ClearField(out vec4 color)
{{
  color = vec4(0.0f);
}}

[include: "scene", "planar_waves", "line_grid"]
void ExtractField(
  uvec2 scene_size,
  uvec2 field_size,
  out vec4 color)
{{
  uvec2 point_idx = uvec2(gl_FragCoord.xy);
  vec2 point_pos = GetGridPointPos(scene_size, field_size, vec2(point_idx));
  
  Complex field_val = Complex(0.0f);
  if(point_idx.y == 0u)
  {
    field_val = GetSceneField(scene_size, point_pos);
  }

  color = vec4(field_val.x, field_val.y, 0.0f, 1.0f);
}}

[include: "complex", "bessel"]
[declaration: "planar_waves"]
{{

  vec2 GetPlanarWaveVec(vec2 dir, vec2 tangent, float linear_wave_vec, float wavelength)
  {
    float wave_vec_len = 2.0f * pi / wavelength;
    float tangent_wave_vec_sqr = wave_vec_len * wave_vec_len - linear_wave_vec * linear_wave_vec;
    return dir * linear_wave_vec + tangent * sqrt(max(tangent_wave_vec_sqr, 0.0f));
  }


  struct PlanarWave
  {
    ComplexVec wave_vec;
    Complex phase_mult;
  };
  //add deconvolution?
  PlanarWave GetPlanarWaveFromHarmonic(vec2 p0, vec2 p1, int harmonic_idx, int size, float wavelength)
  {
    vec2 dir = normalize(p1 - p0);
    vec2 tangent = -vec2(-dir.y, dir.x);
    float l = length(p0 - p1);

    int shifted_idx = ShiftIndex(harmonic_idx, size);
    float wave_vec_1d = float(shifted_idx) * 2.0f * pi / l;

    PlanarWave localWave;
    vec2 wave_vec = GetPlanarWaveVec(dir, tangent, wave_vec_1d, wavelength);

    float wave_vec_len = 2.0f * pi / wavelength;
    float tangent_wave_vec_sqr = wave_vec_len * wave_vec_len - wave_vec_1d * wave_vec_1d;

    PlanarWave planar_wave;
    if(tangent_wave_vec_sqr > 0.0f)
    {
      planar_wave.wave_vec = ComplexVecFromReIm(dir * wave_vec_1d + tangent * sqrt(tangent_wave_vec_sqr), Complex(0.0f));
    }else
    {
      planar_wave.wave_vec = ComplexVecFromReIm(dir * wave_vec_1d, +tangent * sqrt(-tangent_wave_vec_sqr));
    }
    ComplexVec center_delta = ComplexVecFromReIm((p1 - p0) / 2.0f, Complex(0.0f));
    Complex complex_phase = MulI(ComplexDot(planar_wave.wave_vec, center_delta));
    planar_wave.phase_mult = Exp(complex_phase) / float(size);
    if(tangent_wave_vec_sqr <= 0.0f)
    {
      planar_wave.phase_mult *= 0.0f;
      planar_wave.wave_vec = ComplexVecFromReIm(vec2(0.0f), vec2(0.0f));
    }

    return planar_wave;
  }

  float GetOcclusionFactor(vec2 pos, vec2 p0, vec2 p1, ComplexVec wave_vec)
  {
    vec2 real_dir = normalize(vec2(wave_vec.x.x, wave_vec.y.x));
    vec2 tangent = vec2(-real_dir.y, real_dir.x);
    float side0 = dot(pos - p0, tangent);
    float side1 = dot(pos - p1, tangent);
    float res = (side0 * side1 < 0.0f) ? 1.0f : 0.0f;
    return res;
  }

  float IsRightQuadrant(ComplexVec wave_vec)
  {
    vec2 real_dir = (vec2(wave_vec.x.x, wave_vec.y.x));
    //return 1.0f;
    return (abs(real_dir.x) < abs(real_dir.y)) ? 1.0f : 0.0f;
  }

  PlanarWave GetReferenceWave(
    int harmonic_idx,
    vec2 p0,
    vec2 p1,
    sampler2D field_fft,
    int size,
    float wavelength)
  {
    vec2 center = (p0 + p1) * 0.5f;

    PlanarWave planar_wave = GetPlanarWaveFromHarmonic(p0, p1, harmonic_idx, size, wavelength);
    Complex contrib = texelFetch(field_fft, ivec2(harmonic_idx, 0), 0).xy;
    ComplexVec delta0 = ComplexVecFromReIm(-center, Complex(0.0f));

    Complex phase_shift = Exp(MulI(ComplexDot(delta0, planar_wave.wave_vec)));

    PlanarWave res_wave;
    res_wave.wave_vec = planar_wave.wave_vec;
    res_wave.phase_mult = Mul(contrib, Mul(planar_wave.phase_mult, phase_shift));
    return res_wave;
  }

  Complex ReconstructField(
    vec2 pos,
    vec2 p0,
    vec2 p1,
    sampler2D field_fft,
    int size,
    int line_idx,
    int simulate_evanescent_waves,
    int simulate_wave_occlusion,
    float wavelength)
  {
    vec2 center = (p0 + p1) * 0.5f;

    Complex res = Complex(0.0f);
    for(int harmonic_idx = 0; harmonic_idx < int(size); harmonic_idx++)
    {
      PlanarWave planar_wave = GetPlanarWaveFromHarmonic(p0, p1, harmonic_idx, size, wavelength);
      Complex contrib = texelFetch(field_fft, ivec2(harmonic_idx, line_idx), 0).xy;
      ComplexVec delta = ComplexVecFromReIm(pos - center, Complex(0.0f));
      Complex complex_phase = MulI(ComplexDot(delta, planar_wave.wave_vec));

      /*if(simulate_evanescent_waves == 0)
      {
        complex_phase.x = 0.0f;
      }
      complex_phase.x = -abs(complex_phase.x); //should this be needed?*/
      contrib = Mul(contrib, Exp(complex_phase));
      contrib = Mul(contrib, planar_wave.phase_mult);
      if(simulate_wave_occlusion != 0)
        contrib *= GetOcclusionFactor(pos, p0, p1, planar_wave.wave_vec);
      res += contrib;
    }
    return res;
  }
}}

[include: "scene", "planar_waves"]
[blendmode: additive]
void CaptureReferenceWave(
  vec2 field_p0,
  vec2 field_p1,
  sampler2D field_fft_tex,
  int field_size,
  out vec4 color)
{{
  int harmonic_idx = int(gl_FragCoord.x);
  PlanarWave ref_wave = GetReferenceWave(
    harmonic_idx,
    field_p0,
    field_p1,
    field_fft_tex,
    field_size,
    wavelength);
  color = vec4(ref_wave.phase_mult, 0.0f, 0.0f);
}}

[include: "scene", "planar_waves", "line_grid"]
void FinalGatheringShader(
  uvec2 scene_size,
  uvec2 field_size,
  sampler2D field_img,
  out vec4 color)
{{
  Complex ref_field_val = GetSceneField(scene_size, gl_FragCoord.xy);
  vec2 point_idx2f = GetGridPointIdx(scene_size, field_size, gl_FragCoord.xy);
  ivec2 nearest_point_idx = ivec2(floor(point_idx2f + vec2(0.5f)));

  color = vec4(hash3i3f(ivec3(nearest_point_idx, 0)), 1.0f);
  color = vec4(texelFetch(field_img, nearest_point_idx, 0));
}}

[include: "scene", "planar_waves", "line_grid"]
void PropagateField(
  uvec2 scene_size,
  uvec2 field_size,
  sampler2D in_field_fft,
  out vec4 out_field_color)
{{
  ivec2 point_idx = ivec2(gl_FragCoord.xy);
  vec2 point_pos = GetGridPointPos(scene_size, field_size, vec2(point_idx));
  
  int prev_line_idx = point_idx.y - 1;
  if(point_idx.y == 0)
  {
    out_field_color = vec4(GetSceneField(scene_size, point_pos), 0.0f, 1.0f);
  }else
  {
    Complex res_field = ReconstructField(
      point_pos,
      GetGridPointPos(scene_size, field_size, vec2(0.0f, point_idx.y)),
      GetGridPointPos(scene_size, field_size, vec2(float(field_size.x), point_idx.y)),
      in_field_fft,
      int(field_size.x),
      int(prev_line_idx),
      1,
      1,
      wavelength);
    out_field_color = vec4(res_field, 0.0f, 1.0f);
  }
  /*Complex field_val = Complex(0.0f);
  if(point_idx.y == 0)
  {
    field_val = GetSceneField(scene_size, point_pos);
  }

  out_field_color = vec4(field_val.x, field_val.y, 0.0f, 1.0f);*/
  //out_field_color = vec4(0.5f);
}}


[rendergraph]
[include: "fps"]
void RenderGraphMain()
{{
  if(Checkbox("Run"))
  {
    uvec2 scene_size = GetSwapchainImage().GetSize();
    ClearShader(GetSwapchainImage());

    uvec2 field_size = uvec2(
      SliderInt("DFT resolution", 16, 2048, 128),
      SliderInt("Lines count", 1, 128, 2));

    Image curr_field_img = GetImage(field_size, rgba32f);
    Image curr_field_fft = GetImage(field_size, rgba32f);

    if(SliderInt("Init", 0, 1, 1) == 1)
      ExtractField(scene_size, field_size, curr_field_img);

    DFT1(
      curr_field_img,
      field_size,
      1,
      curr_field_fft);

    PropagateField(scene_size, field_size, curr_field_fft, curr_field_img);

    FinalGatheringShader(
      scene_size,
      field_size,
      curr_field_img,
      GetSwapchainImage());


    //OverlayTexShader(
    //  merged_atlas_img,
    //  GetSwapchainImage()); 
  }

  Text("Fps: " + GetSmoothFps());
}}

void ClearShader(out vec4 col)
{{
  col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
}}

void CopyShader(sampler2D tex, out vec4 col)
{{
  col = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}}

[declaration: "pcg"]
{{
  //http://www.jcgt.org/published/0009/03/02/paper.pdf
  uvec3 hash33UintPcg(uvec3 v)
  {
      v = v * 1664525u + 1013904223u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy; //swizzled notation is not exactly the same because components depend on each other, but works too

      v ^= v >> 16u;
      v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
      //v += v.yzx * v.zxy;
      return v;
  }

  vec3 hash3i3f(ivec3 seed)
  {
      uvec3 hash_uvec3 = hash33UintPcg(uvec3(seed));
      return vec3(hash_uvec3) * (1.0f / float(~0u));
  }
}}

[declaration: "smoothing"]
{{
  float SmoothOverTime(float val, string name, float ratio = 0.95)
  {
    ContextVec2(name) = ContextVec2(name) * ratio + vec2(val, 1) * (1.0 - ratio);
    return ContextVec2(name).x / (1e-7f + ContextVec2(name).y);
  }
}}

[declaration: "complex"]
[include: "pi"]
{{
  #define Complex vec2
  Complex ExpI(float theta)
  {
    return Complex(cos(theta), sin(theta));
  }
  Complex Exp(Complex c)
  {
    return exp(c.x) * ExpI(c.y);
  }

  Complex Conjugate(Complex c)
  {
    return Complex(c.x, -c.y);
  }

  Complex Mul(Complex a, Complex b)
  {
    return Complex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  }

  Complex Div(Complex a, Complex b)
  {
    return Mul(a, Conjugate(b)) / dot(b, b);
  }

  Complex SafeInverse(Complex a, float eps)
  {
    return Mul(Complex(1.0f, 0.0f), Conjugate(a)) / max(eps, dot(a, a));
  }
  Complex MulI(Complex a)
  {
    return Complex(-a.y, a.x);
  }

  struct ComplexVec
  {
    Complex x;
    Complex y;
  };

  ComplexVec ComplexVecFromReIm(vec2 re, vec2 im)
  {
    ComplexVec res;
    res.x = Complex(re.x, im.x);
    res.y = Complex(re.y, im.y);
    return res;
  }

  Complex ComplexDot(ComplexVec a, ComplexVec b)
  {
    return Mul(a.x, b.x) + Mul(a.y, b.y);
  }

  int ShiftIndex(int src_index, int size)
  {
    return ((src_index + size / 2) % size) - size / 2;
  }

  int UnShiftIndex(int src_index, int size)
  {
    return (((src_index + size) % size) + size) % size;
  }

  ivec2 ShiftIndex(ivec2 src_index, ivec2 size)
  {
    return ((src_index + size / 2) % size) - size / 2;
  }

  ivec2 UnShiftIndex(ivec2 src_index, ivec2 size)
  {
    return (((src_index + size) % size) + size) % size;
  }
}}

[include: "config", "complex"]
void DFT1(
  sampler2D src_tex,
  uvec2 size,
  uint is_forward,
  out vec4 color)
{{
  float phase_dir = (is_forward == 1u) ? -1.0f : 1.0f;
  float mult = (is_forward == 1u) ? 1.0f : (1.0f / float(size.x));
  vec4 harmonic = vec4(0.0f);

  uint i = uint(gl_FragCoord.x);
  uint y = uint(gl_FragCoord.y);
  for (uint j = 0u; j < size.x; j++)
  {
    vec4 src_pixel = texelFetch(src_tex, ivec2(j, y), 0);
    harmonic += vec4(
      Mul(src_pixel.xy, ExpI(2.0f * phase_dir * pi * float(i * j) / float(size.x))),
      Mul(src_pixel.zw, ExpI(2.0f * phase_dir * pi * float(i * j) / float(size.x))));
  }
  color = harmonic * mult;
}}


[include: "pi"]
[declaration: "bessel"]
{{
  //https://www.shadertoy.com/view/Wt3czM
  // License: CC0 (https://creativecommons.org/publicdomain/zero/1.0/)

  /*
      Approximations for the Bessel functions J0 and J1 and the Struve functions H0 and H1.
      https://en.wikipedia.org/wiki/Bessel_function
      https://en.wikipedia.org/wiki/Struve_function
  */

  // https://link.springer.com/article/10.1007/s40314-020-01238-z
  float BesselJ0(float x)
  {
      float xx = x * x;
      float lamb = 0.865;
      float q    = 0.7172491568;
      float p0   = 0.6312725339;
      float ps0  = 0.4308049446;
      float p1   = 0.3500347951;
      float ps1  = 0.4678202347;
      float p2   =-0.06207747907;
      float ps2  = 0.04253832927;

      float lamb4 = (lamb * lamb) * (lamb * lamb);
      float t0 = sqrt(1.0 + lamb4 * xx);
      float t1 = sqrt(t0);
      
      return xx == 0.0 ? 1.0 : 1.0/(t1 * (1.0 + q * xx)) * ((p0 + p1*xx + p2*t0) * cos(x) + ((ps0 + ps1*xx) * t0 + ps2*xx) * (sin(x)/x));
  }

  // https://www.sciencedirect.com/science/article/pii/S2211379718300111
  float BesselJ1(float x)
  {
      float xx = x * x;

      return (sqrt(1.0 + 0.12138 * xx) * (46.68634 + 5.82514 * xx) * sin(x) - x * (17.83632 + 2.02948 * xx) * cos(x)) /
            ((57.70003 + 17.49211 * xx) * pow(1.0 + 0.12138 * xx, 3.0/4.0) );
  }

  // https://research.tue.nl/nl/publications/efficient-approximation-of-the-struve-functions-hn-occurring-in-the-calculation-of-sound-radiation-quantaties(c68b8858-9c9d-4ff2-bf39-e888bb638527).html
  float StruveH0(float x)
  {
      float xx = x * x;

      return BesselJ1(x) + 1.134817700  * (1.0 - cos(x))/x - 
                          1.0943193181 * (sin(x) - x * cos(x))/xx - 
                          0.5752390840 * (x * 0.8830472903 - sin(x * 0.8830472903))/xx;
  }

  // https://research.tue.nl/nl/publications/efficient-approximation-of-the-struve-functions-hn-occurring-in-the-calculation-of-sound-radiation-quantaties(c68b8858-9c9d-4ff2-bf39-e888bb638527).html
  float StruveH1(float x)
  {
      float xx = x * x;

      return 2.0/pi - BesselJ0(x) + 0.0404983827 * sin(x)/x + 
                                    1.0943193181 * (1.0 - cos(x))/xx - 
                                    0.5752390840 * (1.0 - cos(x * 0.8830472903))/xx;
  }
  // https://www.shadertoy.com/view/tlf3D2
  // evaluate integer-order Bessel function of the first kind using the midpoint rule; https://doi.org/10.1002/sapm1955341298
  // see also https://doi.org/10.2307/2695765 and https://doi.org/10.1137/130932132 for more details
  float besselJ(int n, float x)
  {
    int m = 14;
      float mm = float(m), nn = float(n);
      float s = 0.0, h = 0.5 * pi/mm;
      
      for (int k = 0; k < m; k++)
      {
          float t = h * (float(k) + 0.5);
          s += (((n & 1) == 1) ? (sin(x * sin(t)) * sin(nn * t)) : (cos(x * cos(t)) * cos(nn * t)))/mm;
      }
      
      return ((n & 1) == 1) ? s : (((((n >> 1) & 1) == 1) ? -1.0 : 1.0) * s);
  }
  //https://www.astro.rug.nl/~gipsy/sub/bessel.c
  //File:         bessel.c
  //Author:       M.G.R. Vogelaar
  
   float bessj0( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of first kind and order  */
  /*          0 at input x                                      */
  /*------------------------------------------------------------*/
  {
    float ax,z;
    float xx,y,ans,ans1,ans2;

    if ((ax=abs(x)) < 8.0) {
        y=x*x;
        ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
          +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
        ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
          +y*(59272.64853+y*(267.8532712+y*1.0))));
        ans=ans1/ans2;
    } else {
        z=8.0/ax;
        y=z*z;
        xx=ax-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
          +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
          +y*(-0.6911147651e-5+y*(0.7621095161e-6
          -y*0.934935152e-7)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    }
    return ans;
  }



   float bessj1( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of first kind and order  */
  /*          1 at input x                                      */
  /*------------------------------------------------------------*/
  {
    float ax,z;
    float xx,y,ans,ans1,ans2;

    if ((ax=abs(x)) < 8.0) {
        y=x*x;
        ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
          +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
        ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
          +y*(99447.43394+y*(376.9991397+y*1.0))));
        ans=ans1/ans2;
    } else {
        z=8.0/ax;
        y=z*z;
        xx=ax-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
          +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
          +y*(0.8449199096e-5+y*(-0.88228987e-6
          +y*0.105787412e-6)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
        if (x < 0.0) ans = -ans;
    }
    return ans;
  }

   float bessy0( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of second kind and order */
  /*          0 at input x.                                     */
  /*------------------------------------------------------------*/
  {
    float z;
    float xx,y,ans,ans1,ans2;

    if (x < 8.0) {
        y=x*x;
        ans1 = -2957821389.0+y*(7062834065.0+y*(-512359803.6
          +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
        ans2=40076544269.0+y*(745249964.8+y*(7189466.438
          +y*(47447.26470+y*(226.1030244+y*1.0))));
        ans=(ans1/ans2)+0.636619772*bessj0(x)*log(x);
    } else {
        z=8.0/x;
        y=z*z;
        xx=x-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
          +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
          +y*(-0.6911147651e-5+y*(0.7621095161e-6
          +y*(-0.934945152e-7))));
        ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
    }
    return ans;
  }



   float bessy1( float x )
  /*------------------------------------------------------------*/
  /* PURPOSE: Evaluate Bessel function of second kind and order */
  /*          1 at input x.                                     */
  /*------------------------------------------------------------*/
  {
    float z;
    float xx,y,ans,ans1,ans2;

    if (x < 8.0) {
        y=x*x;
        ans1=x*(-0.4900604943e13+y*(0.1275274390e13
          +y*(-0.5153438139e11+y*(0.7349264551e9
          +y*(-0.4237922726e7+y*0.8511937935e4)))));
        ans2=0.2499580570e14+y*(0.4244419664e12
          +y*(0.3733650367e10+y*(0.2245904002e8
          +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
        ans=(ans1/ans2)+0.636619772*(bessj1(x)*log(x)-1.0/x);
    } else {
        z=8.0/x;
        y=z*z;
        xx=x-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
          +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
          +y*(0.8449199096e-5+y*(-0.88228987e-6
          +y*0.105787412e-6)));
        ans=sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
    }
    return ans;
  }


}}
  
[declaration: "fps"]
[include: "smoothing"]
{{
  float GetSmoothFps()
  {
    float dt = GetTime() - ContextFloat("prev_time");
    ContextFloat("prev_time") = GetTime();

    return 1000.0 / (1e-7f + SmoothOverTime(dt, "fps_count"));
  }
}}
