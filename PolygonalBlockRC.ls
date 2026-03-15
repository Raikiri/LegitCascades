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

  const int cascades_count = 6;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 10;
  uint c0_line_spacing = 10;
  uint c0_dirs_count = SliderInt("c0_dirs_count/4", 1, 4, 1) * 4;

  uint curr_probe_spacing = c0_probe_spacing;
  uint curr_line_spacing = c0_line_spacing;
  uint curr_dirs_count = c0_dirs_count;

  /*for(uint cascade_idx = 0; cascade_idx < cascades_count; cascade_idx++)
  {
    uint curr_lines_count = (viewport_size.x + curr_line_spacing - 1) / curr_line_spacing;
    uint curr_probes_count = (viewport_size.y + curr_probe_spacing - 1) / curr_probe_spacing;

    uvec2 curr_size = uvec2(curr_lines_count * curr_dirs_count * 2, curr_probes_count);
    extended_cascades.insertLast(GetImage(curr_size, rgba16f));
    merged_cascades.insertLast(GetImage(curr_size, rgba16f));
    curr_line_spacing *= 2;
    curr_dirs_count *= 2;
    //Text("c" + to_string(cascade_idx) + " size" +  to_string(curr_size));
  }*/

  LoadCheckerboard(GetSwapchainImage(), c0_probe_spacing);

  float length_scale = SliderFloat("length_scale", 0.0f, 5.0f, 1.0f);
  int test_probe_idx_x = SliderInt("probe_idx_x", 0, 100, 0);
  int test_probe_idx_y = SliderInt("probe_idx_y", 0, 100, 0);
  float shrinkage = SliderFloat("shrinkage", 0.0f, 5.0f, 1.0f);
  int connect_lines = SliderInt("connect_lines", 0, 1, 0);

  ProbeLayoutTestShader(
    c0_probe_spacing,
    c0_dirs_count,
    ivec2(test_probe_idx_x, test_probe_idx_y),
    length_scale,
    shrinkage,
    connect_lines,
    GetSwapchainImage());

  Text("Fps: " + GetSmoothFps());
}}

[include: "config", "pcg", "utils", "block_probes"]
void ProbeLayoutTestShader(
  uint c0_probe_spacing,
  uint c0_dirs_count,
  ivec2 test_probe_idx,
  float length_scale,
  float shrinkage,
  int connect_lines,
  out vec4 color)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  ivec2 tile_idx = pixel_idx / int(c0_probe_spacing);
  color = vec4(0.005f) * GetCheckerboard(tile_idx);

  for(uint cascade_idx = 0u; cascade_idx < 5u; cascade_idx++)
  {
    uint probe_spacing = c0_probe_spacing << cascade_idx;
    uint dirs_count = c0_dirs_count << cascade_idx;
    ivec2 cascade_probe_idx = test_probe_idx >> cascade_idx;
    for(uint dir_idx = 0u; dir_idx < dirs_count; dir_idx++)
    {
      if(IsPointInPolygon(gl_FragCoord.xy, cascade_probe_idx, float(dir_idx), float(probe_spacing), dirs_count, length_scale, connect_lines == 1, shrinkage))
      {
        color += vec4(hash3i3f(ivec3(dir_idx, 0, 0)), 0.0f) * 0.4f;
      }
    }
  }
}}

[include: "utils"]
void LoadCheckerboard(out vec4 col, uint spacing)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  
  col = vec4(vec3(0.001f), 1.0f) * GetCheckerboard(pixel_idx / int(spacing));
}}

void ClearShader(out vec4 col)
{{
  col = vec4(0.0f, 0.0f, 0.0f, 1.0f);
}}

void CopyShader(sampler2D tex, out vec4 col)
{{
  col = texelFetch(tex, ivec2(gl_FragCoord.xy), 0);
}}

[blendmode: additive]
void RenderPoint(uint c0_probe_spacing, vec2 light_pos, out vec4 color)
{{
  vec2 light_pixel_pos = light_pos * vec2(c0_probe_spacing);
  color = vec4(0.0f);
  if(length(light_pixel_pos - gl_FragCoord.xy) < 2.0f)
  {
    color = vec4(1.0f);
  }
}}


[include: "aabb"]
[declaration: "block_probes"]
{{
  vec4 GetProbeInnerAabb(ivec2 probe_idx, float probe_spacing, float length_scale)
  {
    return vec4(vec2(probe_idx) - vec2(1.0f) * length_scale, vec2(probe_idx) + vec2(1.0f + length_scale)) * probe_spacing;
  }

  vec4 GetProbeOuterAabb(ivec2 probe_idx, float probe_spacing, float length_scale)
  {
    ivec2 parent_probe_idx = probe_idx / 2;
    float parent_probe_spacing = probe_spacing * 2.0f;
    return GetProbeInnerAabb(parent_probe_idx, parent_probe_spacing, length_scale);
  }

  struct Line
  {
    vec2 points[2];
  };
  Line GetProbeLineConnected(ivec2 probe_idx, float dir_idxf, float probe_spacing, uint dirs_count, float length_scale)
  {
    vec4 inner_aabb = GetProbeInnerAabb(probe_idx, probe_spacing, length_scale);
    vec4 outer_aabb = GetProbeOuterAabb(probe_idx, probe_spacing, length_scale);

    float dir_ratio = (dir_idxf + 0.5f) / float(dirs_count);

    vec2 norm_points[2];
    norm_points[0] = GetNormAabbPerimeterPoint(dir_ratio);
    norm_points[1] = GetNormAabbPerimeterPoint(dir_ratio);

    Line probe_line;
    probe_line.points[0] = mix(inner_aabb.xy, inner_aabb.zw, norm_points[0]);
    probe_line.points[1] = mix(outer_aabb.xy, outer_aabb.zw, norm_points[1]);
    return probe_line;
  }
  Line GetProbeLineDisconnected(ivec2 probe_idx, float dir_idxf, float probe_spacing, uint dirs_count, float length_scale)
  {
    vec4 inner_aabb = GetProbeInnerAabb(probe_idx, probe_spacing, length_scale);
    vec4 outer_aabb = GetProbeOuterAabb(probe_idx, probe_spacing, length_scale);

    float dir_ratio = (dir_idxf + 0.5f) / float(dirs_count);

    vec2 norm_point;
    norm_point = GetNormAabbPerimeterPoint(dir_ratio);
    vec2 ray_dir = normalize(norm_point - vec2(0.5f));

    Line probe_line;
    probe_line.points[0] = mix(inner_aabb.xy, inner_aabb.zw, norm_point);

    vec2 t = RayAabbIntersect(outer_aabb.xy, outer_aabb.zw, probe_line.points[0], ray_dir);
    probe_line.points[1] = probe_line.points[0] + ray_dir * t.y;
    return probe_line;
  }

  bool IsPointInPolygon(vec2 point, ivec2 probe_idx, float dir_idx, float probe_spacing, uint dirs_count, float length_scale, bool connect_lines, float margin)
  {
    Line min_probe_line;
    Line max_probe_line;
    if(connect_lines)
    {
      min_probe_line = GetProbeLineConnected(probe_idx, float(dir_idx) - 0.5f, probe_spacing, dirs_count, length_scale);
      max_probe_line = GetProbeLineConnected(probe_idx, float(dir_idx) + 0.5f, probe_spacing, dirs_count, length_scale);
    }else
    {
      min_probe_line = GetProbeLineDisconnected(probe_idx, float(dir_idx) - 0.5f, probe_spacing, dirs_count, length_scale);
      max_probe_line = GetProbeLineDisconnected(probe_idx, float(dir_idx) + 0.5f, probe_spacing, dirs_count, length_scale);
    }

    vec4 inner_aabb = GetProbeInnerAabb(probe_idx, float(probe_spacing), length_scale);
    vec4 outer_aabb = GetProbeOuterAabb(probe_idx, float(probe_spacing), length_scale);
    return
      PointLineDist(point, min_probe_line.points[0], min_probe_line.points[1]) > margin &&
      PointLineDist(point, max_probe_line.points[0], max_probe_line.points[1]) < -margin &&
      !IsPointInAabb(point, vec4(inner_aabb.xy - vec2(margin), inner_aabb.zw + vec2(margin))) &&
       IsPointInAabb(point, vec4(outer_aabb.xy + vec2(margin), outer_aabb.zw - vec2(margin)));
  }


  float GetProbeDirIdxf(ivec2 probe_idx, vec2 ray_origin, vec2 ray_dir, float probe_spacing, uint dirs_count, float debug_scale)
  {
    vec4 inner_aabb = GetProbeInnerAabb(probe_idx, probe_spacing, debug_scale);
    vec2 t = RayAabbIntersect(inner_aabb.xy, inner_aabb.zw, ray_origin, ray_dir);
    vec2 p = ray_origin + ray_dir * t.y;

    vec2 norm_point = (p - inner_aabb.xy) / (inner_aabb.zw - inner_aabb.xy);
    float dir_ratio = GetNormAabbPerimeterRatio(norm_point);
    return dir_ratio * float(dirs_count) - 0.5f;
  }
}}

[declaration: "aabb"]
{{
  vec2 SafeInv(vec2 v)
  {
    float large_val = 1e7f;
    return vec2(
      abs(v.x) > 0.0f ? 1.0f / v.x : large_val,
      abs(v.y) > 0.0f ? 1.0f / v.y : large_val
    );
  }
  vec2 RayAabbIntersect(vec2 aabb_min, vec2 aabb_max, vec2 ray_origin, vec2 ray_dir)
  {
    vec2 inv_dir = SafeInv(ray_dir);
    vec2 t1 = (aabb_min - ray_origin) * inv_dir;
    vec2 t2 = (aabb_max - ray_origin) * inv_dir;

    vec2 t;
    t.x = max(min(t1.x, t2.x), min(t1.y, t2.y));
    t.y = min(max(t1.x, t2.x), max(t1.y, t2.y));
    return t;
  }
  
  bool IsPointInAabb(vec2 point, vec4 aabb_minmax)
  {
    return
      point.x >= aabb_minmax.x &&
      point.x <= aabb_minmax.z &&
      point.y >= aabb_minmax.y &&
      point.y <= aabb_minmax.w;
  }

  vec2 GetNormAabbPerimeterPoint(float ratio)
  {
    float perimeter_coord = ratio * 4.0f;
    int side_idx = int(floor(perimeter_coord)) % 4;
    float side_ratio = fract(perimeter_coord);

    if(side_idx == 0) return vec2(side_ratio, 0.0f);
    if(side_idx == 1) return vec2(1.0f, side_ratio);
    if(side_idx == 2) return vec2(1.0f - side_ratio, 1.0f);
    return vec2(0.0f, 1.0f - side_ratio);
  }

  float GetNormAabbPerimeterRatio(vec2 point)
  {
    float eps = 1e-2f;
    if(point.y < eps)
      return (point.x + 0.0f) / 4.0f;
    if(point.x > 1.0f - eps)
      return (point.y + 1.0f) / 4.0f;
    if(point.y > 1.0f - eps)
      return ((1.0f - point.x) + 2.0f) / 4.0f;
    return ((1.0f - point.y) + 3.0f) / 4.0f;
  }
}}
[declaration: "utils"]
{{
  float GetCheckerboard(ivec2 p)
  {
    return ((p.x + p.y) % 2 == 0) ? 0.0f : 1.0f;
  }
  float PointEdgeDist(vec2 p, vec2 p0, vec2 p1)
  {
      vec2 delta = p1 - p0;
      float scale = dot(p - p0, delta) / dot(delta, delta);
      vec2 proj = p0 + delta * scale;
      return scale > 0.0f && scale < 1.0f ? length(proj - p) : min(length(p - p0), length(p - p1));
      //return length(proj - p);
  }
  float cross2(vec2 v0, vec2 v1)
  {
    return v0.x * v1.y - v0.y * v1.x;
  }
  bool PointIsInConvex(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3)
  {
    bool s0 = cross2(p1 - p0, p - p0) > 0.0f;
    bool s1 = cross2(p2 - p1, p - p1) > 0.0f;
    bool s2 = cross2(p3 - p2, p - p2) > 0.0f;
    bool s3 = cross2(p0 - p3, p - p3) > 0.0f;
    return s0 && s1 && s2 && s3;
  }
  float PointLineDist(vec2 p, vec2 p0, vec2 p1)
  {
    vec2 delta = p1 - p0;
    vec2 perp = vec2(-delta.y, delta.x);
    return dot(p - p0, normalize(perp));
  }
  bool PointIsInConvexMargin(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3, float margin)
  {
    
    bool s0 = PointLineDist(p, p0, p1) > margin;
    bool s1 = PointLineDist(p, p1, p2) > margin;
    bool s2 = PointLineDist(p, p2, p3) > margin;
    bool s3 = PointLineDist(p, p3, p0) > margin;
    return s0 && s1 && s2 && s3;
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
