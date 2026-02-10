// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/EnergyConservingHRC.ls

[declaration: "config"]
{{
  #define POS_FIRST_LAYOUT 1
  const float pi = 3.141592f;
}}

[declaration: "block_probe_layout"]
{{
  struct IntervalIdx
  {
    ivec2 probe_idx;
    ivec2 point_ids;
  };

  IntervalIdx AtlasTexelIdxToIntervalIdx(ivec2 atlas_texel_idx, uint probe_points_count)
  {
    uvec2 block_size = uvec2(probe_points_count * 4u);
    IntervalIdx interval_idx;
    interval_idx.probe_idx = atlas_texel_idx / int(block_size);
    interval_idx.point_ids = atlas_texel_idx % int(block_size);
    return interval_idx;
  }

  vec2 GetNormAABBPerimeterPoint(float ratio)
  {
    float perimeter_coord = ratio * 4.0f;
    int side_idx = clamp(int(floor(perimeter_coord)), 0, 3);
    float side_ratio = fract(perimeter_coord);

    if(side_idx == 0) return vec2(side_ratio, 0.0f);
    if(side_idx == 1) return vec2(1.0f, side_ratio);
    if(side_idx == 2) return vec2(1.0f - side_ratio, 1.0f);
    return vec2(0.0f, 1.0f - side_ratio);
  }

  vec4 IntervalIdxToIntervalPoints(ivec2 probe_idx, ivec2 point_ids, uint probe_points_count, uint probe_spacing)
  {
    vec2 ratio = (vec2(point_ids) + vec2(0.5f)) / (4.0f * float(probe_points_count));
    vec4 norm_points = vec4(GetNormAABBPerimeterPoint(ratio.x), GetNormAABBPerimeterPoint(ratio.y));
    vec4 probe_minmax = vec4(vec2(probe_idx), vec2(probe_idx + ivec2(1))) * float(probe_spacing);
    return vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_points.zw
    );
  }
}}
[include: "pcg", "block_probe_layout"]
void FinalGatheringShader(
  out vec4 color)
{{
  color = vec4(1.0f, 0.5f, 0.0f, 1.0f);
}}

[rendergraph]
[include: "fps"]
void RenderGraphMain()
{{
  uvec2 viewport_size = GetSwapchainImage().GetSize();
  ClearShader(GetSwapchainImage());
  uvec2 c0_atlas_size = viewport_size;
  //c0_size.x = SliderInt("c0_size.x", 1, 1024, 256);
  //c0_size.y = size.y * c0_size.x / size.x;
  vec2 light_pixel_pos;
  light_pixel_pos.x = SliderFloat("light_pixel_pos.x", 0.0f, 512.0f, 100.0f);
  light_pixel_pos.y = SliderFloat("light_pixel_pos.y", 0.0f, 512.0f, 100.0f);
  float light_pixel_height = SliderFloat("light_pixel_height", 0.0f, 512.0f, 100.0f);

  const int cascades_count = 6;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 10;
  uint c0_probe_points_count = 1;

  uint curr_probe_spacing = c0_probe_spacing;
  uint curr_probe_points_count = c0_probe_points_count;

  uvec2 c0_probes_count = viewport_size / c0_probe_spacing;
  for(uint cascade_idx = 0; cascade_idx < 3; cascade_idx++)
  {
    uvec2 curr_probes_count = viewport_size / curr_probe_spacing;
    uvec2 curr_size = curr_probes_count * curr_probe_points_count * 4;
    extended_cascades.insertLast(GetImage(curr_size, rgba16f));
    merged_cascades.insertLast(GetImage(curr_size, rgba16f));
    curr_probe_spacing *= 2;
    curr_probe_points_count *= 2;
    Text("c" + to_string(cascade_idx) + " size" +  to_string(curr_size));
  }

  FinalGatheringShader(
    GetSwapchainImage()
  );
  /*OverlayTexShader(
    dst_merged_img,
    GetSwapchainImage());*/

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

[declaration: "scene"]
{{
  vec4 Circle(vec4 prev_radiance, vec2 delta, float radius, vec4 circle_radiance)
  {
    return length(delta) < radius ? circle_radiance : prev_radiance;
  }
}}

[include: "scene"]
void SceneShader(uvec2 size, out vec4 radiance)
{{
  radiance = vec4(0.0f);
  radiance = Circle(radiance, vec2(size / 2u) - gl_FragCoord.xy, 30.0, vec4(1.0f, 0.5f, 0.0f, 1.0f));
  //radiance = Circle(radiance, vec2(size / 2u) + vec2(10.0, 0.0) - gl_FragCoord.xy, 30.0, vec4(0.1f, 0.1f, 0.1f, 1.0f));
  radiance = Circle(radiance, vec2(size / 2u) + vec2(150.0, 50.0) - gl_FragCoord.xy, 30.0, vec4(0.0f, 0.0f, 0.0f, 1.0f));
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
