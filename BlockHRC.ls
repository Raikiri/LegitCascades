// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/EnergyConservingHRC.ls

[declaration: "config"]
{{
  #define POS_FIRST_LAYOUT 1
}}

[declaration: "geometry_utils"]
{{
  const float pi = 3.141592f;
  float PointEdgeDist(vec2 p, vec2 p0, vec2 p1)
  {
      vec2 delta = p1 - p0;
      float scale = dot(p - p0, delta) / dot(delta, delta);
      vec2 proj = p0 + delta * scale;
      return scale > 0.0f && scale < 1.0f ? length(proj - p) : min(length(p - p0), length(p - p1));
      //return length(proj - p);
  }

  bool RayAABBIntersect(vec4 aabb_minmax, vec2 ray_origin, vec2 ray_dir, out float tmin, out float tmax)
  {
    vec2 t1 = (aabb_minmax.xy - ray_origin) / ray_dir;
    vec2 t2 = (aabb_minmax.zw - ray_origin) / ray_dir;

    tmin = max(min(t1.x, t2.x), min(t1.y, t2.y));
    tmax = min(max(t1.x, t2.x), max(t1.y, t2.y));
    return tmax >= tmin;
  }

  float cross2d(vec2 v0, vec2 v1)
  {
    return v0.x * v1.y - v0.y * v1.x;
  }
}}

[declaration: "hrc_basis"]
{{

  vec2 GetEdgeAngRange(vec2 p, vec4 edge)
  {
      vec2 delta0 = edge.xy - p;
      vec2 delta1 = edge.zw - p;
      return vec2(atan(delta0.y, delta0.x), atan(delta1.y, delta1.x));
  }

  vec4 SplitWrappedRange(vec2 range)
  {
      if(range.y < range.x) return vec4(range.x, 1.0f, 0.0f, range.y);
      return vec4(range.xy, 0.0f, 0.0f);
  }

  float GetIntervalOverlap(vec2 range0, vec2 range1)
  {
      return max(0.0f, min(range0.y, range1.y) - max(range0.x, range1.x));
  }

  float WrappedIntersection(vec2 range0, vec2 range1)
  {
      vec4 split_range0 = SplitWrappedRange(fract(range0));
      vec4 split_range1 = SplitWrappedRange(fract(range1));
      
      return
          GetIntervalOverlap(split_range0.xy, split_range1.xy) +
          GetIntervalOverlap(split_range0.xy, split_range1.zw) +
          GetIntervalOverlap(split_range0.zw, split_range1.zw) +
          GetIntervalOverlap(split_range0.zw, split_range1.xy);
  }

  float GetBasisAnalytical(vec2 p, vec4 edge0, vec4 edge1)
  {
      vec2 range0 = GetEdgeAngRange(p, edge0) / (2.0f * 3.1415f);
      vec2 range1 = GetEdgeAngRange(p, edge1) / (2.0f * 3.1415f);
      
      if(fract(range0.y - range0.x) > 0.5f) range0 = range0.yx;
      if(fract(range1.y - range1.x) > 0.5f) range1 = range1.yx;
      
      return max(WrappedIntersection(range0, range1), WrappedIntersection(range0 + vec2(0.5f), range1));
  }
}}

[declaration: "block_probe_layout"]
[include: "geometry_utils"]
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

  ivec2 IntervalIdxToAtlasTexelIdx(ivec2 probe_idx, ivec2 point_ids, uint probe_points_count)
  {
    uvec2 block_size = uvec2(probe_points_count * 4u);
    return ivec2(block_size) * probe_idx + point_ids;
  }

  vec2 GetNormAABBPerimeterPoint(float ratio)
  {
    float perimeter_coord = ratio * 4.0f;
    int side_idx = int(floor(perimeter_coord)) % 4;
    float side_ratio = fract(perimeter_coord);

    if(side_idx == 0) return vec2(side_ratio, 0.0f);
    if(side_idx == 1) return vec2(1.0f, side_ratio);
    if(side_idx == 2) return vec2(1.0f - side_ratio, 1.0f);
    return vec2(0.0f, 1.0f - side_ratio);
  }

  float GetNormAAABBPerimeterRatio(vec2 point)
  {
    float eps = 1e-5f;
    if(point.y < eps)
      return point.x + 0.0f;
    if(point.x > 1.0f - eps)
      return point.y + 1.0f;
    if(point.y > 1.0f - eps)
      return (1.0f - point.x) + 2.0f;
    return (1.0f - point.y) + 3.0f;
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

  vec4 GetProbePointEdge(ivec2 probe_idx, int point_idx, uint probe_points_count, uint probe_spacing)
  {
    vec2 edge_ratio = vec2(point_idx, point_idx + 1) / (4.0f * float(probe_points_count));

    vec4 norm_edge_points = vec4(GetNormAABBPerimeterPoint(edge_ratio.x), GetNormAABBPerimeterPoint(edge_ratio.y));

    vec4 probe_minmax = vec4(vec2(probe_idx), vec2(probe_idx + ivec2(1))) * float(probe_spacing);
    return vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge_points.zw
    );
  }

  vec2 RayToPointIdsf(ivec2 probe_idx, vec2 ray_origin, vec2 ray_dir, uint probe_points_count, uint probe_spacing)
  {
    vec4 probe_minmax = vec4(vec2(probe_idx), vec2(probe_idx + ivec2(1))) * float(probe_spacing);
    vec2 t;
    bool is_hit = RayAABBIntersect(probe_minmax, ray_origin, ray_dir, t.x, t.y);
    if(!is_hit || t.x < -1e-5f) return vec2(-1.0f);

    vec2 point_idsf;
    for(uint i = 0u; i < 2u; i++)
    {
      vec2 inter_point = ray_origin + ray_dir * t[i];
      vec2 norm_pos = (inter_point - probe_minmax.xy) / float(probe_spacing);

      point_idsf[i] = GetNormAAABBPerimeterRatio(norm_pos);
    }
    return point_idsf;
  }

  uint GetProbeSpacing(uint c0_probe_spacing, uint cascade_idx)
  {
    return c0_probe_spacing << cascade_idx;
  }
  uint GetProbePointsCount(uint c0_probe_points_count, uint cascade_idx)
  {
    return c0_probe_points_count << cascade_idx;
  }
}}

[include: "pcg", "block_probe_layout"]
void ExtendCascade(
  uint c0_probe_spacing,
  uint c0_probe_points_count,
  uint dst_cascade_idx,
  sampler2D src_cascade_atlas,
  out vec4 color)
{{
  uint src_cascade_idx = dst_cascade_idx - 1u;


  uint src_probe_spacing = GetProbeSpacing(c0_probe_spacing, src_cascade_idx);
  uint src_probe_points_count = GetProbePointsCount(c0_probe_points_count, src_cascade_idx);

  uint dst_probe_spacing = GetProbeSpacing(c0_probe_spacing, dst_cascade_idx);
  uint dst_probe_points_count = GetProbePointsCount(c0_probe_points_count, dst_cascade_idx);

  ivec2 dst_atlas_texel = ivec2(gl_FragCoord.xy);
  IntervalIdx dst_interval_idx = AtlasTexelIdxToIntervalIdx(dst_atlas_texel, dst_probe_points_count);

  vec4 dst_edge0 = GetProbePointEdge(dst_interval_idx.probe_idx, dst_interval_idx.point_ids.x, dst_probe_points_count, dst_probe_spacing);
  vec4 dst_edge1 = GetProbePointEdge(dst_interval_idx.probe_idx, dst_interval_idx.point_ids.y, dst_probe_points_count, dst_probe_spacing);



  color = vec4(0.0f, 0.0f, 0.0f, 0.0f);

  //if(dst_interval_idx.point_ids.x == 3 && dst_interval_idx.point_ids.y == 6)
  {
    for(uint y_offset = 0u; y_offset < 2u; y_offset++)
    {
      for(uint x_offset = 0u; x_offset < 2u; x_offset++)
      {
        //ivec2 src_probe_idx = ivec2(4, 3);
        ivec2 src_probe_idx = dst_interval_idx.probe_idx * 2 + ivec2(x_offset, y_offset);
        
        uint count = 10u;
        for(uint y = 0u; y < count; y++)
        {
          for(uint x = 0u; x < count; x++)
          {
            vec2 ratio = (vec2(x, y) + vec2(0.5f)) / float(count);
            vec2 ray_start = mix(dst_edge0.xy, dst_edge0.zw, ratio.x);
            vec2 ray_end = mix(dst_edge1.xy, dst_edge1.zw, ratio.y);
            vec2 ray_dir = normalize(ray_end - ray_start);
            vec2 dir0 = normalize(dst_edge0.zw - dst_edge0.xy);
            vec2 dir1 = normalize(dst_edge1.zw - dst_edge1.xy);
            float w = abs(asin(cross2d(dir0, ray_dir))) * abs(asin(cross2d(dir1, ray_dir)));
            vec2 src_point_idsf = RayToPointIdsf(src_probe_idx, ray_start, ray_end - ray_start, src_probe_points_count, src_probe_spacing);
            if(src_point_idsf.x > 0.0f)
            {
              ivec2 src_point_ids = ivec2(floor(src_point_idsf));
              ivec2 src_atlas_texel_idx = IntervalIdxToAtlasTexelIdx(src_probe_idx, src_point_ids, src_probe_points_count);

              color += texelFetch(src_cascade_atlas, src_atlas_texel_idx, 0) / float(count * count);
            }
          }
        }
      }
    }
  }

  //if(dst_interval_idx.point_ids.y == 6) color = vec4(1.0f, 0.5f, 0.0f, 1.0f);
}}

[include: "pcg", "block_probe_layout", "geometry_utils", "hrc_basis"]
void FinalGatheringShader(
  uint c0_probe_spacing,
  uint c0_probe_points_count,
  uint cascade_idx,
  sampler2D cascade_atlas,
  out vec4 color)
{{
  vec2 pixel_pos = gl_FragCoord.xy;
  uint probe_spacing = GetProbeSpacing(c0_probe_spacing, cascade_idx);
  uint probe_points_count = GetProbePointsCount(c0_probe_points_count, cascade_idx);
  vec2 probe_idxf = pixel_pos / vec2(probe_spacing);
  //ivec2 probe_idx = ivec2(floor(probe_idxf));
  ivec2 probe_idx = ivec2(7, 5) / int(1u << cascade_idx);

  color = vec4(0.0f);
  for(uint point_idx0 = 0u; point_idx0 < probe_points_count * 4u; point_idx0++)
  {
    for(uint point_idx1 = 0u; point_idx1 < probe_points_count * 4u; point_idx1++)
    {
      ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(probe_idx, ivec2(point_idx0, point_idx1), probe_points_count);
      vec4 interval_points = IntervalIdxToIntervalPoints(probe_idx, ivec2(point_idx0, point_idx1), probe_points_count, probe_spacing);

      float edge_dist = PointEdgeDist(pixel_pos, interval_points.xy, interval_points.zw);

      if(point_idx0 != point_idx1)
      {
        vec4 edge0 = GetProbePointEdge(probe_idx, int(point_idx0), probe_points_count, probe_spacing);
        vec4 edge1 = GetProbePointEdge(probe_idx, int(point_idx1), probe_points_count, probe_spacing);
        float basis_func = GetBasisAnalytical(pixel_pos, edge0, edge1);
        color += basis_func * texelFetch(cascade_atlas, atlas_texel_idx, 0);
      }
    }
  }

  /*uint count = 100u;
  for(uint dir_idx = 0u; dir_idx < count; dir_idx++)
  {
    float ratio = (float(dir_idx) + 0.5f) / float(count);
    float ang = ratio * pi * 2.0f;
    vec2 ray_dir = vec2(cos(ang), sin(ang));
    vec2 point_idxf = RayToPointIdsf(probe_idx, pixel_pos, ray_dir, probe_points_count, probe_spacing);
    if(point_idxf.x > 3.0f && point_idxf.x < 3.99f && point_idxf.y > 0.0f && point_idxf.y < 1.0f)
    {
      color += vec4(1.0f, 0.5f, 0.0f, 0.0f) / float(count);
    }
  }*/



  //ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(probe_idx, ivec2(1, 3), probe_points_count);
  //color += texelFetch(cascade_atlas, atlas_texel_idx, 0);
}}

[include: "block_probe_layout"]
void SetCascade(
  uint c0_probe_spacing,
  uint c0_probe_points_count,
  uint cascade_idx,
  out vec4 atlas_texel)
{{
  ivec2 atlas_texel_idx = ivec2(gl_FragCoord.xy);
  uint probe_points_count = GetProbePointsCount(c0_probe_points_count, cascade_idx);
  uvec2 probe_spacing = uvec2(GetProbeSpacing(c0_probe_spacing, cascade_idx));

  IntervalIdx interval_idx = AtlasTexelIdxToIntervalIdx(atlas_texel_idx, probe_points_count);
  atlas_texel = vec4(0.0f, 0.0f, 0.0f, 1.0f);

  if((interval_idx.probe_idx.x == 7) && (interval_idx.probe_idx.y == 5))
  {
    atlas_texel = vec4(1.0f, 0.5f, 0.0f, 0.0f) * 0.5f;
  }
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

  const int cascades_count = 6;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 50;
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

  SetCascade(
    c0_probe_spacing,
    c0_probe_points_count,
    0,
    extended_cascades[0]
  );

  for(uint src_cascade_idx = 0; src_cascade_idx < 2; src_cascade_idx++)
  {
    ExtendCascade(
      c0_probe_spacing,
      c0_probe_points_count,
      src_cascade_idx + 1,
      extended_cascades[src_cascade_idx],
      extended_cascades[src_cascade_idx + 1]
    );
  }

  /*FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    0,
    extended_cascades[0],
    GetSwapchainImage()
  );*/
  int gather_cascade_idx = SliderInt("Gather cascade_idx", 0, 2, 0);

  FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    gather_cascade_idx,
    extended_cascades[gather_cascade_idx],
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
