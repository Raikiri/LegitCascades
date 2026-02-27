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

    uvec2 curr_size = uvec2(curr_lines_count * curr_dirs_count, curr_probes_count);
    extended_cascades.insertLast(GetImage(curr_size, rgba16f));
    merged_cascades.insertLast(GetImage(curr_size, rgba16f));
    curr_line_spacing *= 2;
    curr_dirs_count *= 2;
    Text("c" + to_string(cascade_idx) + " size" +  to_string(curr_size));
  }

  ivec2 source_tile_idx;
  source_tile_idx.x = SliderInt("Source x", 0, 256, 14);
  source_tile_idx.y = SliderInt("Source y", 0, 256, 7);
  ProbeLayoutTestShader(
    c0_probe_spacing,
    c0_line_spacing,
    c0_dirs_count,
    source_tile_idx, GetSwapchainImage());

  Text("Fps: " + GetSmoothFps());
}}

[include: "config", "pcg", "utils", "polygon_layout"]
void ProbeLayoutTestShader(
  uint c0_probe_spacing,
  uint c0_line_spacing,
  uint c0_dirs_count,
  ivec2 source_tile_idx,
  out vec4 color)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  ivec2 tile_idx = pixel_idx / int(c0_probe_spacing);
  color = vec4(0.005f) * GetCheckerboard(tile_idx);

  uint cascade_idx = 0u;
  uint probe_spacing = c0_probe_spacing;
  uint line_spacing = c0_line_spacing << cascade_idx;

  int test_line_idx = 3;
  float test_probe_idx = 0.0f;
  float test_dir_idx = 0.0f;
  float probe_func = GetProbeFunction(
    gl_FragCoord.xy,
    test_line_idx,
    test_probe_idx,
    test_dir_idx,
    probe_spacing,
    line_spacing);

  color += vec4(0.0f, 1.0f, 0.0f, 0.0f) * 0.1f * probe_func;
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
  float GetPolygonFunction(vec2 pos, float left_x, vec2 left_y_range, float right_x, vec2 right_y_range)
  {
    if(pos.x >= left_x && pos.x < right_x)
    {
      //return 1.0f;
      float x_ratio = (pos.x - left_x) / (right_x - left_x);
      vec2 y_range = mix(left_y_range, right_y_range, x_ratio);
      if(pos.y >= y_range.x && pos.y < y_range.y) return 1.0f;
    }
    return 0.0f;
  }

  float GetProbeFunction(
    vec2 pos,
    int line_idx,
    float probe_idxf,
    float dir_idxf,
    uint probe_spacing,
    uint line_spacing)
  {
    float left_line_x = float(line_idx) * float(line_spacing);
    vec2 left_line_y_range = (vec2(probe_idxf) + vec2(0.0f, 1.0f)) * float(probe_spacing);
    bool is_extended_line = (line_idx % 2) == 0;
    float right_line_x = float(line_idx + (is_extended_line ? 2 : 1)) * float(line_spacing);

    vec2 right_line_y_range = (vec2(probe_idxf) + (vec2(dir_idxf) + vec2(0.0f, 1.0f)) * 2.0f) * float(probe_spacing);
    return GetPolygonFunction(pos, left_line_x, left_line_y_range, right_line_x, right_line_y_range);
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
