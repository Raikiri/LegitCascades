// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/EnergyConservingHRC.ls

[declaration: "config"]
{{
  #define POS_FIRST_LAYOUT 1
  const float pi = 3.141592f;
}}

[declaration: "geometry_utils"]
{{
  float PointEdgeDist(vec2 p, vec2 p0, vec2 p1)
  {
      vec2 delta = p1 - p0;
      float scale = dot(p - p0, delta) / dot(delta, delta);
      vec2 proj = p0 + delta * scale;
      return scale > 0.0f && scale < 1.0f ? length(proj - p) : min(length(p - p0), length(p - p1));
      //return length(proj - p);
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

  struct BasisEdges
  {
    vec4 edge0;
    vec4 edge1;
  };

  BasisEdges IntervalIdxToIntervalEdges(ivec2 probe_idx, ivec2 point_ids, uint probe_points_count, uint probe_spacing)
  {
    vec2 edge0_ratio = vec2(point_ids.x, point_ids.x + 1) / (4.0f * float(probe_points_count));
    vec2 edge1_ratio = vec2(point_ids.y, point_ids.y + 1) / (4.0f * float(probe_points_count));

    vec4 norm_edge0_points = vec4(GetNormAABBPerimeterPoint(edge0_ratio.x), GetNormAABBPerimeterPoint(edge0_ratio.y));
    vec4 norm_edge1_points = vec4(GetNormAABBPerimeterPoint(edge1_ratio.x), GetNormAABBPerimeterPoint(edge1_ratio.y));

    vec4 probe_minmax = vec4(vec2(probe_idx), vec2(probe_idx + ivec2(1))) * float(probe_spacing);
    BasisEdges edges;
    edges.edge0 = vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge0_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge0_points.zw
    );
    edges.edge1 = vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge1_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge1_points.zw
    );
    return edges;
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


  uvec2 src_probe_spacing = uvec2(GetProbeSpacing(c0_probe_spacing, src_cascade_idx));
  uint src_probe_points_count = GetProbePointsCount(c0_probe_points_count, src_cascade_idx);

  uvec2 dst_probe_spacing = uvec2(GetProbeSpacing(c0_probe_spacing, dst_cascade_idx));
  uint dst_probe_points_count = GetProbePointsCount(c0_probe_points_count, dst_cascade_idx);

  ivec2 dst_atlas_texel = ivec2(gl_FragCoord.xy);
  IntervalIdx dst_interval_idx = AtlasTexelIdxToIntervalIdx(dst_atlas_texel, dst_probe_points_count);

  color = vec4(0.0f, 0.0f, 0.0f, 0.0f);
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
  ivec2 probe_idx = ivec2(floor(probe_idxf));
  //ivec2 probe_idx = ivec2(4, 3);

  color = vec4(0.0f);
  for(uint point_idx0 = 0u; point_idx0 < probe_points_count * 4u; point_idx0++)
  {
    for(uint point_idx1 = 0u; point_idx1 < probe_points_count * 4u; point_idx1++)
    {
      ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(probe_idx, ivec2(point_idx0, point_idx1), probe_points_count);
      vec4 interval_points = IntervalIdxToIntervalPoints(probe_idx, ivec2(point_idx0, point_idx1), probe_points_count, probe_spacing);

      float edge_dist = PointEdgeDist(pixel_pos, interval_points.xy, interval_points.zw);

      /*if(edge_dist < 2.0f)
      {
        color += texelFetch(cascade_atlas, atlas_texel_idx, 0);
      }*/

      if(point_idx0 != point_idx1)
      {
        BasisEdges edges = IntervalIdxToIntervalEdges(probe_idx, ivec2(point_idx0, point_idx1), probe_points_count, probe_spacing);
        float basis_func = GetBasisAnalytical(pixel_pos, edges.edge0, edges.edge1);
        color += basis_func * texelFetch(cascade_atlas, atlas_texel_idx, 0);
      }
    }
  }


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

  if((interval_idx.probe_idx.x == 4) && (interval_idx.probe_idx.y == 3))
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
  vec2 light_pixel_pos;
  light_pixel_pos.x = SliderFloat("light_pixel_pos.x", 0.0f, 512.0f, 100.0f);
  light_pixel_pos.y = SliderFloat("light_pixel_pos.y", 0.0f, 512.0f, 100.0f);
  float light_pixel_height = SliderFloat("light_pixel_height", 0.0f, 512.0f, 100.0f);

  const int cascades_count = 6;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 100;
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

  /*ExtendCascade(
    c0_probe_spacing,
    c0_probe_points_count,
    1,
    extended_cascades[0],
    extended_cascades[1]
  );*/

  FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    0,
    extended_cascades[0],
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
