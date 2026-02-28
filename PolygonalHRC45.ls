// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/BlockHRC.ls

[declaration: "config"]
{{
}}

[rendergraph]
[include: "fps", "config"]
void RenderGraphMain()
{{
  uvec2 viewport_size = GetSwapchainImage().GetSize();
  ClearShader(GetSwapchainImage());
  uvec2 c0_atlas_size = viewport_size;
  //c0_size.x = SliderInt("c0_size.x", 1, 1024, 256);
  //c0_size.y = size.y * c0_size.x / size.x;

  const int cascades_count = 9;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 30;
  uint c0_line_spacing = 30;
  uint c0_dirs_count = 1;

  uint curr_probe_spacing = c0_probe_spacing;
  uint curr_line_spacing = c0_line_spacing;
  uint curr_dirs_count = c0_dirs_count;

  for(uint cascade_idx = 0; cascade_idx < cascades_count; cascade_idx++)
  {
    uint curr_lines_count = (viewport_size.x + curr_line_spacing - 1) / curr_line_spacing;
    uint curr_probes_count = (viewport_size.y + curr_probe_spacing - 1) / curr_probe_spacing;

    uvec2 curr_size = uvec2(curr_lines_count * curr_dirs_count * 2, curr_probes_count);
    extended_cascades.insertLast(GetImage(curr_size, rgba16f));
    merged_cascades.insertLast(GetImage(curr_size, rgba16f));
    curr_line_spacing *= 2;
    curr_dirs_count *= 2;
    //Text("c" + to_string(cascade_idx) + " size" +  to_string(curr_size));
  }

  uint test_line_idx = SliderInt("line_idx", 0, 32, 0);
  uint test_probe_idx = SliderInt("probe_idx", 0, 16, 0);
  uint test_probes_count = SliderInt("probes_count", 1, 16, 1);

  float frustum_shrinkening = SliderFloat("shrinkage", 0.0f, 2.0f, 1.0f);
  float cascade_tint = SliderFloat("tint", 0.0f, 1.0f, 1.0f);

  LoadCascade(
    c0_probe_spacing,
    c0_line_spacing,
    c0_dirs_count,
    0,
    test_line_idx,
    test_probe_idx,
    0,
    test_probes_count,
    extended_cascades[0]);
  LoadCheckerboard(GetSwapchainImage(), c0_probe_spacing);
  GatherCascade(
    c0_probe_spacing,
    c0_line_spacing,
    c0_dirs_count,
    viewport_size,
    0,
    frustum_shrinkening,
    cascade_tint,
    extended_cascades[0],
    GetSwapchainImage());

  uint used_cascades_count = SliderInt("cascades_count", 1, cascades_count, 6);
  for(uint src_cascade_idx = 0; src_cascade_idx < used_cascades_count - 1; src_cascade_idx++)
  {
    ExtendCascade(
      c0_probe_spacing,
      c0_line_spacing,
      c0_dirs_count,
      viewport_size,
      src_cascade_idx,
      extended_cascades[src_cascade_idx],
      extended_cascades[src_cascade_idx + 1]);


    GatherCascade(
      c0_probe_spacing,
      c0_line_spacing,
      c0_dirs_count,
      viewport_size,
      src_cascade_idx + 1,
      frustum_shrinkening,
      cascade_tint,
      extended_cascades[src_cascade_idx + 1],
      GetSwapchainImage());
  }

  /*int test_cascade_idx = SliderInt("cascade_idx", 0, 10, 0);
  int test_dir_idx = SliderInt("dir_idx", 0, 10, 0);
  int test_is_frustum = SliderInt("is_frustum", 0, 1, 1);

  ProbeLayoutTestShader(
    c0_probe_spacing,
    c0_line_spacing,
    test_cascade_idx,
    test_line_idx,
    test_probe_idx,
    test_dir_idx,
    test_is_frustum,
    GetSwapchainImage());*/

  Text("Fps: " + GetSmoothFps());
}}

[include: "config", "pcg", "utils", "polygon_layout"]
void ExtendCascade(
  uint c0_probe_spacing,
  uint c0_line_spacing,
  uint c0_dirs_count,
  uvec2 viewport_size,
  uint src_cascade_idx,
  sampler2D src_cascade_atlas_tex,
  out vec4 dst_color)
{{
  ivec2 dst_atlas_texel_idx = ivec2(gl_FragCoord.xy);

  uint dst_cascade_idx = src_cascade_idx + 1u;
  uint dst_dirs_count = c0_dirs_count << dst_cascade_idx;
  PolygonIdx dst_polygon_idx = AtlasTexelIdxToPolygonIdx(dst_atlas_texel_idx, dst_dirs_count);

  uint src_dirs_count = c0_dirs_count << src_cascade_idx;
  vec4 res_radiance = vec4(0.0f);

  if(dst_polygon_idx.is_frustum)
  {
    {
      int src_line_idx = dst_polygon_idx.line_idx * 2 - 2;
      {
        if((dst_polygon_idx.dir_idx % 2) == 0)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - dst_polygon_idx.dir_idx;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, true, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
        if((dst_polygon_idx.dir_idx % 2) == 1)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - (dst_polygon_idx.dir_idx + 1);

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, true, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
      }
    }
    {
      int src_line_idx = dst_polygon_idx.line_idx * 2 - 1;
      {
        if((dst_polygon_idx.dir_idx % 2) == 0)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - src_dir_idx;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, true, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
        if((dst_polygon_idx.dir_idx % 2) == 1)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - src_dir_idx - 1;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, true, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
      }
    }
  }else
  {
    {
      int src_line_idx = dst_polygon_idx.line_idx * 2 - 2;
      {
        if(dst_polygon_idx.dir_idx % 2 == 1)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - dst_polygon_idx.dir_idx - 0;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, true, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }

        if(dst_polygon_idx.dir_idx % 2 == 0)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - dst_polygon_idx.dir_idx;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, false, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
      }
    }
    {
      int src_line_idx = dst_polygon_idx.line_idx * 2 - 1;
      {
        if(dst_polygon_idx.dir_idx % 2 == 0)
        {
          int src_dir_idx = dst_polygon_idx.dir_idx / 2;
          int src_probe_idx = dst_polygon_idx.probe_idx - src_dir_idx;

          if(src_dir_idx >= 0 && src_dir_idx < int(src_dirs_count))
          {
            ivec2 src_texel_idx = PolygonIdxToAtlasTexelIdx(src_line_idx, src_probe_idx, src_dir_idx, false, src_dirs_count);
            vec4 src_radiance = texelFetch(src_cascade_atlas_tex, src_texel_idx, 0);
            res_radiance += src_radiance;
          }
        }
      }
    }
  }

  dst_color = res_radiance;
}}

[include: "config", "pcg", "utils", "polygon_layout"]
[blendmode: additive]
void GatherCascade(
  uint c0_probe_spacing,
  uint c0_line_spacing,
  uint c0_dirs_count,
  uvec2 viewport_size,
  uint cascade_idx,
  float frustum_shrinkage,
  float cascade_tint_amount,
  sampler2D cascade_atlas_tex,
  out vec4 color)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  color = vec4(0.0f);

  uint probe_spacing = c0_probe_spacing;
  uint probes_count = viewport_size.y / probe_spacing;
  uint line_spacing = c0_line_spacing << cascade_idx;
  uint lines_count = viewport_size.x / probe_spacing;
  uint dirs_count = c0_dirs_count << cascade_idx;

  vec4 cascade_tint = vec4(mix(vec3(1.0f), hash3i3f(ivec3(cascade_idx, 2, 3)), cascade_tint_amount), 1.0f);
  float dir_tint_amount = 0.0f;//0.5f;
  vec2 falloff_range = GetCascadeFalloffRange(cascade_idx);

  vec4 fluence = vec4(0.0f);

  int line_idx = pixel_idx.x / int(line_spacing);
  for(int probe_idx = 0; probe_idx < int(probes_count); probe_idx++)
  {
    for(int dir_idx = 0; dir_idx < int(dirs_count); dir_idx++)
    {
      for(int is_frustum = 0; is_frustum < 2; is_frustum++)
      {
        ivec2 texel_idx = PolygonIdxToAtlasTexelIdx(line_idx, probe_idx, dir_idx, is_frustum == 1, dirs_count);
        float probe_func = GetProbeFunction(
          gl_FragCoord.xy,
          line_idx,
          float(probe_idx),
          float(dir_idx),
          is_frustum == 1,
          probe_spacing,
          line_spacing,
          frustum_shrinkage);
        if(probe_func > -0.5f)
        {
          float cascade_radiance = mix(falloff_range.x, falloff_range.y, probe_func);
          vec4 cascade_texel = texelFetch(cascade_atlas_tex, texel_idx, 0);
          float dir_tint = ((dir_idx % 2) == 0 ? 1.0f : 0.0f) * dir_tint_amount + (1.0f - dir_tint_amount);
          fluence += cascade_tint * dir_tint * cascade_radiance * cascade_texel;
        }
      }
    }
  }
  if(line_idx % 2 == 1)
  {
    int extended_line_idx = line_idx - 1;
    for(int probe_idx = 0; probe_idx < int(probes_count); probe_idx++)
    {
      for(int dir_idx = 0; dir_idx < int(dirs_count); dir_idx++)
      {
        for(int is_frustum = 0; is_frustum < 2; is_frustum++)
        {
          ivec2 texel_idx = PolygonIdxToAtlasTexelIdx(extended_line_idx, probe_idx, dir_idx, is_frustum == 1, dirs_count);
          float probe_func = GetProbeFunction(
            gl_FragCoord.xy,
            extended_line_idx,
            float(probe_idx),
            float(dir_idx),
            is_frustum == 1,
            probe_spacing,
            line_spacing,
            frustum_shrinkage);
          if(probe_func > -0.5f)
          {
            float cascade_radiance = mix(falloff_range.x, falloff_range.y, probe_func);
            vec4 cascade_texel = texelFetch(cascade_atlas_tex, texel_idx, 0);
            float dir_tint = ((dir_idx % 2) == 0 ? 1.0f : 0.0f) * dir_tint_amount + (1.0f - dir_tint_amount);
            fluence += cascade_tint * dir_tint * cascade_radiance * cascade_texel;
          }
        }
      }
    }
  }
  color = fluence;
}}

[include: "config", "polygon_layout"]
void LoadCascade(
  uint c0_probe_spacing,
  uint c0_line_spacing,
  uint c0_dirs_count,
  uint test_cascade_idx,
  int test_line_idx,
  int test_probe_idx,
  int test_dir_idx,
  int test_probes_count,
  out vec4 color)
{{
  uint dirs_count = c0_dirs_count << test_cascade_idx;
  ivec2 atlas_texel_idx = ivec2(gl_FragCoord.xy);
  PolygonIdx polygon_idx = AtlasTexelIdxToPolygonIdx(atlas_texel_idx, dirs_count);

  color = vec4(0.0f);
  if(polygon_idx.line_idx == test_line_idx && polygon_idx.probe_idx >= test_probe_idx && polygon_idx.probe_idx < test_probe_idx + test_probes_count && polygon_idx.dir_idx == test_dir_idx && polygon_idx.is_frustum)
  {
    color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  }
}}

[include: "config", "pcg", "utils", "polygon_layout"]
void ProbeLayoutTestShader(
  uint c0_probe_spacing,
  uint c0_line_spacing,
  uint test_cascade_idx,
  int test_line_idx,
  int test_probe_idx,
  int test_dir_idx,
  int test_is_frustum,
  out vec4 color)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  ivec2 tile_idx = pixel_idx / int(c0_probe_spacing);
  color = vec4(0.005f) * GetCheckerboard(tile_idx);

  uint probe_spacing = c0_probe_spacing;
  uint line_spacing = c0_line_spacing << test_cascade_idx;


  float probe_func = GetProbeFunction(
    gl_FragCoord.xy,
    test_line_idx,
    float(test_probe_idx),
    float(test_dir_idx),
    test_is_frustum == 1,
    probe_spacing,
    line_spacing,
    1.0f);
  
  vec2 falloff_range = GetCascadeFalloffRange(test_cascade_idx);
  if(probe_func >= -0.5f)
  {
    color += vec4(0.0f, 1.0f, 0.0f, 0.0f) * 0.1f * mix(falloff_range.x, falloff_range.y, probe_func);
  }
}}

[include: "utils"]
void LoadCheckerboard(out vec4 col, uint spacing)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  
  col = vec4(vec3(0.005f), 1.0f) * GetCheckerboard(pixel_idx / int(spacing));
}}

void ClearShader(out vec4 col)
{{
  col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
}}

void CopyShader(sampler2D tex, out vec4 col)
{{
  col = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}}

[declaration: "polygon_layout"]
{{
  vec2 GetCascadeFalloffRange(uint cascade_idx)
  {
    return vec2(1.0f) / vec2(float(1u << cascade_idx), float(1u << (cascade_idx + 1u)));
  }
  float GetPolygonFunction(vec2 pos, float left_x, vec2 left_y_range, float right_x, vec2 right_y_range)
  {
    if(pos.x >= left_x && pos.x < right_x)
    {
      float x_ratio = (pos.x - left_x) / (right_x - left_x);
      vec2 y_range = mix(left_y_range, right_y_range, x_ratio);
      if(pos.y >= y_range.x && pos.y < y_range.y)
      {
        return x_ratio;
      }
    }
    return -1.0f;
  }

  float GetProbeFunction(
    vec2 pos,
    int line_idx,
    float probe_idxf,
    float dir_idxf,
    bool is_frustum,
    uint probe_spacing,
    uint line_spacing,
    float frustum_shrinkage)
  {
    float left_line_x = float(line_idx) * float(line_spacing);
    vec2 left_line_y_range = (vec2(probe_idxf) + vec2(0.0f, 1.0f)) * float(probe_spacing) + vec2(frustum_shrinkage, -frustum_shrinkage);
    bool is_extended_line = (line_idx % 2) == 0;
    float right_line_x = float(line_idx + (is_extended_line ? 2 : 1)) * float(line_spacing);

    vec2 y_widening = is_frustum ? vec2(0.0f, is_extended_line ? 3.0f : 2.0f) : vec2(0.0f, 1.0f);
    vec2 right_line_y_range = (vec2(probe_idxf) + vec2(dir_idxf) * (is_extended_line ? 2.0f : 1.0f) + y_widening) * float(probe_spacing) + vec2(frustum_shrinkage, -frustum_shrinkage);
    return GetPolygonFunction(pos, left_line_x, left_line_y_range, right_line_x, right_line_y_range);
  }

  struct PolygonIdx
  {
    int line_idx;
    int probe_idx;
    int dir_idx;
    bool is_frustum;
  };

  PolygonIdx AtlasTexelIdxToPolygonIdx(ivec2 texel_idx, uint dirs_count)
  {
    PolygonIdx polygon_idx;
    polygon_idx.line_idx = texel_idx.x / int(dirs_count * 2u);
    polygon_idx.probe_idx = texel_idx.y;
    polygon_idx.dir_idx = texel_idx.x % int(dirs_count);
    polygon_idx.is_frustum = ((texel_idx.x % int(dirs_count * 2u)) < int(dirs_count));
    return polygon_idx;
  }

  ivec2 PolygonIdxToAtlasTexelIdx(int line_idx, int probe_idx, int dir_idx, bool is_frustum, uint dirs_count)
  {
    return ivec2(line_idx * int(dirs_count * 2u) + (is_frustum ? 0 : int(dirs_count)) + dir_idx, probe_idx);
  }
}}
[declaration: "utils"]
{{
  float GetCheckerboard(ivec2 p)
  {
    return ((p.x + p.y) % 2 == 0) ? 0.0f : 1.0f;
  }
}}
[declaration: "merging"]
{{
  vec4 MergeIntervals(vec4 near_interval, vec4 far_interval)
  {
      //return near_interval + far_interval;
      return vec4(near_interval.rgb + near_interval.a * far_interval.rgb, near_interval.a * far_interval.a);
  }
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

[declaration: "bilinear_interpolation"]
{{
  struct BilinearSamples
  {
      ivec2 base_idx;
      vec2 ratio;
  };

  vec4 GetBilinearWeights(vec2 ratio)
  {
    return vec4(
        (1.0f - ratio.x) * (1.0f - ratio.y),
        ratio.x * (1.0f - ratio.y),
        (1.0f - ratio.x) * ratio.y,
        ratio.x * ratio.y);
  }

  ivec2 GetBilinearOffset(uint offset_index)
  {
    ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));
    return offsets[offset_index];
  }
  BilinearSamples GetBilinearSamples(vec2 pixel_idx2f)
  {
    BilinearSamples samples;
    samples.base_idx = ivec2(floor(pixel_idx2f));
    samples.ratio = fract(pixel_idx2f);
    return samples;
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
