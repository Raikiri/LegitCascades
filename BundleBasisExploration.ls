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

  Image cascade_img = GetImage(viewport_size, rgba16f);

  int source_x = SliderInt("source_x", 0, viewport_size.x, 100);
  int source_y = SliderInt("source_y", 0, viewport_size.y, 100);
  int source_size = SliderInt("source_size", 0, 200, 100);

  vec4 source_minmax = vec4(source_x, source_y - source_size / 2, source_x, source_y + source_size / 2);
  LoadCascade(source_minmax, cascade_img);
  RenderEdge(source_minmax, vec4(1.0f, 0.5f, 0.0f, 0.0f), GetSwapchainImage());
  BasisTestShader(
    cascade_img,
    GetSwapchainImage());
  /*float length_scale = SliderFloat("length_scale", 0.0f, 5.0f, 1.0f);
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
    GetSwapchainImage());*/

  Text("Fps: " + GetSmoothFps());
}}

[include : "utils"]
void RenderEdge(
  vec4 edge_minmax,
  vec4 edge_color,
  out vec4 color
)
{{
  color = PointEdgeDist(gl_FragCoord.xy, edge_minmax.xy, edge_minmax.zw) < 2.0f ? edge_color : vec4(0.0f);
}}
[include : "hrc_basis"]
void LoadCascade(
  vec4 source_minmax,
  out vec4 interval_radiance
)
{{
  ivec2 func_idx = ivec2(gl_FragCoord.xy);
  uvec2 func_count = GetBasisFuncCount();
  interval_radiance = vec4(1.0f, 0.5f, 0.0f, 0.0f);
  if(func_idx.x < int(func_count.x) && func_idx.y < int(func_count.y))
  {
    LineBasis line_basis = GetLineBasis(vec2(func_idx));
    interval_radiance = vec4(0.0f);

    uint steps_count = 50u;
    uvec2 step_idx;
    for(step_idx.y = 0u; step_idx.y < steps_count; step_idx.y++)
    {
      for(step_idx.x = 0u; step_idx.x < steps_count; step_idx.x++)
      {
        vec2 uv = (vec2(step_idx) + vec2(0.5f)) / float(steps_count);
        Ray ray = UvToRayLineBasis(line_basis.origin_range, line_basis.delta_range, uv);

        vec2 t = EdgeEdgeIntersectParam(source_minmax, vec4(ray.origin, ray.origin + ray.dir));
        if(t.x >= 0.0f && t.x < 1.0f)
        {
          interval_radiance += vec4(1.0f, 0.5f, 0.0f, 0.0f) / float(steps_count * steps_count);
        }
      }
    }

  }
}}

[include: "hrc_basis", "bilinear_interpolation"]
[blendmode: additive]
void BasisTestShader(
  sampler2D cascade_img,  
  out vec4 color
)
{{
  uvec2 step_idx;

  vec4 origin_range = vec4(300.0f, 100.0f, 300.0f, 150.0f);
  vec4 delta_range = vec4(1.0f, -0.1f, 1.0f, 0.1f);

  color = vec4(0.0f);


  ivec2 func_idx;
  for(func_idx.x = 0; func_idx.x < int(GetBasisFuncCount().x); func_idx.x++)
  {
    {
      LineBasis line_basis = GetLineBasis(vec2(func_idx.x, 0.0f));
      if(
        PointEdgeDist(gl_FragCoord.xy, line_basis.origin_range.xy + vec2(0.0f, 4.0f), line_basis.origin_range.zw + vec2(0.0f, -4.0f)) < 1.0f)
      {
        color += vec4(0.0f, 1.0f, 0.0f, 0.0f);
      }
    }

    for(func_idx.y = -1; func_idx.y < int(GetBasisFuncCount().y); func_idx.y++)
    {
      mat4 colors;
      for(uint offset_idx = 0u; offset_idx < 4u; offset_idx++)
      {
        vec4 node_color = vec4(0.0f);
        ivec2 sample_idx = func_idx + ivec2(GetBilinearOffset(offset_idx));
        if(sample_idx.x >= 0 && sample_idx.x < int(GetBasisFuncCount().x) && sample_idx.y >= 0 && sample_idx.y < int(GetBasisFuncCount().y))
        //if(sample_idx.x == 1 && sample_idx.y == 0)
          node_color = texelFetch(cascade_img, sample_idx, 0);
        colors[offset_idx] = node_color;
      }
      LineBasis line_basis = GetLineBasis(vec2(func_idx) + vec2(0.5f));
      vec4 weights = GetLineBasisBilinearWeights(line_basis.origin_range, line_basis.delta_range, gl_FragCoord.xy, 30u);
      color += colors * weights * 3.0f;
    }
  }

  /*const uint steps_count = 900u;
  for(uint step_idx = 0u; step_idx < steps_count; step_idx++)
  {
    float ratio = (float(step_idx) + 0.5f) / float(steps_count);
    float ang = ratio * 3.141592f * 2.0f;
    vec2 ray_dir = vec2(cos(ang), sin(ang));
    vec2 ray_origin = gl_FragCoord.xy;
    vec2 uv = RayToUvLineBasis(origin_range, delta_range, gl_FragCoord.xy, ray_dir);
    vec4 ray_color = colors * GetBilinearWeights(uv);

    if(uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f)
    {
      //color += (length(uv - rec_uv) < 0.01f ? vec4(0.0f, 1.0f, 0.0f, 0.0f) : vec4(1.0f, 0.0f, 0.0f, 0.0f)) / float(steps_count * steps_count);
      color += ray_color / float(steps_count);
    }
  }*/

  //color = vec4(1.0f, 0.5f, 0.0f, 0.0f);
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

[declaration: "hrc_basis"]
[include: "utils", "bilinear_interpolation"]
{{
  struct Ray
  {
    vec2 origin;
    vec2 dir;
  };

  Ray UvToRayLineBasis(vec4 origin_range, vec4 delta_range, vec2 uv)
  {
    Ray res_ray;
    res_ray.origin = mix(origin_range.xy, origin_range.zw, uv.x);
    res_ray.dir = mix(delta_range.xy, delta_range.zw, uv.y);
    return res_ray;
  }

  vec2 RayToUvLineBasis(vec4 origin_range, vec4 delta_range, vec2 ray_origin, vec2 ray_dir)
  {
    return vec2(
      EdgeEdgeIntersectParam(origin_range, vec4(ray_origin, ray_origin + ray_dir)).x,
      EdgeEdgeIntersectParam(ray_origin.xyxy + delta_range, vec4(ray_origin, ray_origin + ray_dir)).x
    );
  }

  vec4 GetLineBasisBilinearWeights(vec4 origin_range, vec4 delta_range, vec2 p, const uint steps_count)
  {
    vec4 weights = vec4(0.0f);
    uvec2 step_idx;
    for(step_idx.y = 0u; step_idx.y < steps_count; step_idx.y++)
    {
      for(step_idx.x = 0u; step_idx.x < steps_count; step_idx.x++)
      {
        vec2 uv = (vec2(step_idx) + vec2(0.5f)) / float(steps_count);
        Ray ray = UvToRayLineBasis(origin_range, delta_range, uv);

        if(abs(PointLineDist(p, ray.origin, ray.origin + ray.dir)) < 1.0f)
        {
          weights += GetBilinearWeights(uv) / float(steps_count * steps_count);
        }
      }
    }
    return weights;
  }

  struct LineBasis
  {
    vec4 origin_range;
    vec4 delta_range;
  };

  uvec2 GetBasisFuncCount()
  {
    return uvec2(20u, 10u);
  }
  LineBasis GetLineBasis(vec2 func_idxf)
  {
    float probe_spacing = 20.0f;
    LineBasis line_basis;
    line_basis.origin_range = vec4(300.0f, func_idxf.x * probe_spacing, 300.0f, (func_idxf.x + 1.0f) * probe_spacing);
    //float delta_step = 2.0f / float(GetBasisFuncCount().y - 1u);
    float delta_step = 2.0f / float(GetBasisFuncCount().y);
    line_basis.delta_range = vec4(1.0f, -1.0f + delta_step * func_idxf.y, 1.0f, -1.0f + delta_step * (func_idxf.y + 1.0f));
    return line_basis;
  }
}}

[declaration: "bilinear_interpolation"]
{{
  struct BilinearSamples
  {
    ivec2 base_index;
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
  BilinearSamples GetBilinearSamples(vec2 pixel_index2f)
  {
    BilinearSamples samples;
    samples.base_index = ivec2(floor(pixel_index2f));
    samples.ratio = fract(pixel_index2f);
    return samples;
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
  vec2 EdgeEdgeIntersectParam(vec4 edge0, vec4 edge1)
  {
    //edge0.xy + (edge0.zw - edge0.xy) * t.x == edge1.xy + (edge1.zw - edge1.xy) * t.y
    vec2 d0 = edge0.zw - edge0.xy;
    vec2 d1 = edge1.zw - edge1.xy;
    vec2 r = edge1.xy - edge0.xy;
    mat2 m = mat2(dot(d0, d0), dot(d0, d1), -dot(d1, d0), -dot(d1, d1));
    return inverse(m) * vec2(dot(r, d0), dot(r, d1));
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
