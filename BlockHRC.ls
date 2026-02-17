// Available at
// https://radiance-cascades.github.io/LegitScriptEditor/?gh=Raikiri/LegitCascades/BlockHRC.ls

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

  vec2 SafeInv(vec2 dir)
  {
    return vec2(dir.x == 0.0f ? 1e7f : 1.0f / dir.x, dir.y == 0.0f ? 1e7f : 1.0f / dir.y);
  }

  bool RayAABBIntersect(vec4 aabb_minmax, vec2 ray_origin, vec2 ray_dir, out float tmin, out float tmax)
  {
    vec2 inv_dir = SafeInv(ray_dir);
    vec2 t1 = (aabb_minmax.xy - ray_origin) * inv_dir;
    vec2 t2 = (aabb_minmax.zw - ray_origin) * inv_dir;

    tmin = max(min(t1.x, t2.x), min(t1.y, t2.y));
    tmax = min(max(t1.x, t2.x), max(t1.y, t2.y));
    return tmax >= tmin;
  }

  float cross2(vec2 v0, vec2 v1)
  {
    return v0.x * v1.y - v0.y * v1.x;
  }

  vec2 SolveQuadratic(float A, float B, float C)
  {
      float D = B * B - 4.0f * A * C;
      if(D < 0.0f) return vec2(1e7f);
      float sqrtD = sqrt(D);
      return (vec2(-B) + vec2(-sqrtD, sqrtD)) / (2.0f * A);
  }

  vec2 RaySphereIntersect(vec2 ray_origin, vec2 ray_dir, vec2 center, float radius)
  {
      //(ray_origin + ray_dir * t - center) ^ 2 = radius ^ 2
      float A = dot(ray_dir, ray_dir);
      float B = 2.0f * dot(ray_dir, ray_origin - center);
      float C = dot(ray_origin - center, ray_origin - center) - radius * radius;
      return SolveQuadratic(A, B, C);
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
      
      return max(WrappedIntersection(range0, range1), WrappedIntersection(range0 + vec2(0.5f), range1)) * 2.0f;
  }

  struct RaySample
  {
      vec2 ray_start;
      vec2 ray_end;
      float weight;
  };
  RaySample SampleBundleRay(vec4 edge0, vec4 edge1, vec2 ratio)
  {
      RaySample ray_sample;
      ray_sample.ray_start = mix(edge0.xy, edge0.zw, ratio.x);
      ray_sample.ray_end = mix(edge1.xy, edge1.zw, ratio.y);

      vec2 dir = normalize(ray_sample.ray_end - ray_sample.ray_start);

      float sin0 = abs(cross2(dir, normalize(edge0.zw - edge0.xy)));
      float sin1 = abs(cross2(dir, normalize(edge1.zw - edge1.xy)));

      ray_sample.weight = sin0 * sin1;// * length(edge0.zw - edge0.xy) * length(edge1.zw - edge1.xy) / (length(ray_sample.ray_end - ray_sample.ray_start) * 3.1415f);
      return ray_sample;
  }
}}

[declaration: "linespaces"]
[include: "geometry_utils"]
{{
  struct Ray
  {
    vec2 origin;
    vec2 dir;
  };


  vec2 GetCirclePoint(float ratio)
  {
    float ang = ratio * 2.0f * pi;
    return vec2(0.5f) + vec2(cos(ang), sin(ang)) * sqrt(2.0f) / 2.0f;
  }

  float GetCircleRatio(vec2 point)
  {
    vec2 delta = point - vec2(0.5f);
    float ang = atan(delta.y, delta.x);
    return fract(ang / (2.0f * pi));
  }

  Ray LineCoordToRay(vec2 line_coord)
  {
    float start_quadrant = floor(line_coord.x * 4.0f);
    vec2 p0 = GetCirclePoint(line_coord.x);
    //vec2 p1 = GetCirclePoint(fract((start_quadrant + 1.0f) / 4.0f + line_coord.y * 3.0f / 4.0f));
    vec2 p1 = GetCirclePoint(line_coord.y);
    
    Ray ray;
    ray.origin = p0;
    ray.dir = p1 - p0;
    
    return ray;
  }

  vec2 RayToLineCoord(vec2 origin, vec2 dir)
  {
    vec2 t = RaySphereIntersect(origin, dir, vec2(0.5f), sqrt(2.0f) / 2.0f);
    if(t.x < 1e7f)
    {
      vec2 p0 = origin + dir * t.x;
      vec2 p1 = origin + dir * t.y;
      vec2 line_coord;
      line_coord.x = GetCircleRatio(p0);
      line_coord.y = GetCircleRatio(p1);

      //float start_quadrant = floor(line_coord.x * 4.0f);
      //line_coord.y = fract((GetCircleRatio(p1) - (start_quadrant + 1.0f) / 4.0f)) * 4.0f / 3.0f;
      return line_coord;
    }else
    {
      return vec2(1e7f);
    }
  }

  float FindLineCoordY(vec2 norm_ray_point, float line_coord_x)
  {
    float start_quadrant = floor(line_coord_x * 4.0f);
    vec2 p0 = GetCirclePoint(line_coord_x);
    vec2 norm_ray_dir = norm_ray_point - p0;
    vec2 t = RaySphereIntersect(p0, norm_ray_dir, vec2(0.5f), sqrt(2.0f) / 2.0f);
    if(t.x < 1e7f)
    {
      vec2 p1 = p0 + norm_ray_dir * t.y;
      return GetCircleRatio(p1);
      //return fract((GetCircleRatio(p1) - (start_quadrant + 1.0f) / 4.0f)) * 4.0f / 3.0f;
    }else
    {
      return -1.0f;
    }
  }
}}

[declaration: "block_probe_layout2"]
[include: "linespaces"]
{{
  struct IntervalIdx
  {
    ivec2 block_idx;
    ivec2 line_idx;
  };

  IntervalIdx AtlasTexelIdxToIntervalIdx(ivec2 atlas_texel_idx, uint block_lines_count2)
  {
    uvec2 block_size = uvec2(block_lines_count2);
    IntervalIdx interval_idx;
    interval_idx.block_idx = atlas_texel_idx / int(block_size);
    interval_idx.line_idx = atlas_texel_idx % int(block_size);
    return interval_idx;
  }

  ivec2 IntervalIdxToAtlasTexelIdx(ivec2 block_idx, ivec2 line_idx, uint block_lines_count2)
  {
    uvec2 block_size = uvec2(block_lines_count2);
    return ivec2(block_size) * block_idx + line_idx;
  }  
  
  Ray LineIdxfToRay(ivec2 block_idx, vec2 line_idxf, uint block_lines_count2, uint probe_spacing)
  {
    vec2 line_coord = (line_idxf + vec2(0.5f)) / float(block_lines_count2);
    Ray norm_ray = LineCoordToRay(line_coord);
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    
    Ray ray;
    ray.origin = probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_ray.origin;
    ray.dir = norm_ray.dir;
    return ray;
  }

  vec2 RayToLineIdxf(ivec2 block_idx, vec2 ray_origin, vec2 ray_dir, uint block_lines_count2, uint probe_spacing)
  {
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    Ray norm_ray;
    norm_ray.origin = (ray_origin - probe_minmax.xy) / float(probe_spacing);
    norm_ray.dir = ray_dir;

    vec2 line_coord = RayToLineCoord(norm_ray.origin, norm_ray.dir);
    if(line_coord.x < 1e7f)
      return line_coord * float(block_lines_count2) - vec2(0.5f);
    else
      return vec2(1e7f);
  }

  float FindLineIdxY(ivec2 block_idx, float line_idx_x, vec2 ray_point, uint block_lines_count2, uint probe_spacing)
  {
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    vec2 norm_ray_point = (ray_point - probe_minmax.xy) / float(probe_spacing);
    return FindLineCoordY(norm_ray_point, (line_idx_x + 0.5f) / float(block_lines_count2)) * float(block_lines_count2) - 0.5f;
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


[include: "pcg", "block_probe_layout2", "hrc_basis", "bilinear_interpolation"]
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

  color = vec4(0.0f, 0.0f, 0.0f, 0.0f);

  vec2 ratio;
  float step_size = 0.1f;
  for(ratio.y = 0.0f; ratio.y < 1.0f; ratio.y += step_size)
  {
    for(ratio.x = 0.0f; ratio.x < 1.0f; ratio.x += step_size)
    {
      Ray dst_ray = LineIdxfToRay(dst_interval_idx.block_idx, vec2(dst_interval_idx.line_idx) - vec2(0.5f) + ratio, dst_probe_points_count, dst_probe_spacing);
      for(uint y_offset = 0u; y_offset < 2u; y_offset++)
      {
        for(uint x_offset = 0u; x_offset < 2u; x_offset++)
        {
          //ivec2 src_probe_idx = ivec2(4, 3);
          ivec2 src_probe_idx = dst_interval_idx.block_idx * 2 + ivec2(x_offset, y_offset);  
          vec2 src_line_idx = RayToLineIdxf(src_probe_idx, dst_ray.origin, dst_ray.dir, src_probe_points_count, src_probe_spacing);
          //vec2 test_dst_line_idx = RayToLineIdxf(dst_interval_idx.block_idx, dst_ray.origin, dst_ray.dir, dst_probe_points_count, dst_probe_spacing);

          //if(abs(float(dst_interval_idx.line_idx.x) - 1.0f) < 5.01f)
          //if(dst_interval_idx.line_idx.x == 2 && dst_interval_idx.line_idx.y == 13)
          vec4 src_probe_minmax = vec4(vec2(src_probe_idx), vec2(src_probe_idx + ivec2(1))) * float(src_probe_spacing);

          vec2 src_probe_t;
          bool src_probe_hit = RayAABBIntersect(src_probe_minmax, dst_ray.origin, dst_ray.dir, src_probe_t.x, src_probe_t.y);
          //if(src_line_idx.y < 1e7f)
          if(src_probe_hit)
          //if(length(test_dst_line_idx - vec2(dst_interval_idx.line_idx)) < 0.01f)
          //if(test_dst_line_idx.x < 1e7f)
          //if(length(dst_ray.dir) != 0.0f)
          {
            BilinearSamples bilinear_samples = GetBilinearSamples(src_line_idx);
            vec4 weights = GetBilinearWeights(bilinear_samples.ratio);
            for(uint sample_idx = 0u; sample_idx < 4u; sample_idx++)
            {
              ivec2 src_line_idx = (bilinear_samples.base_idx + GetBilinearOffset(sample_idx) + ivec2(src_probe_points_count)) % ivec2(src_probe_points_count);
              //ivec2 src_line_idx = bilinear_samples.base_idx + GetBilinearOffset(sample_idx);
              //if(src_line_idx.x >= 0 && src_line_idx.y >= 0 && src_line_idx.x < int(src_probe_points_count) && src_line_idx.y < int(src_probe_points_count))
              {
                ivec2 src_atlas_texel_idx = IntervalIdxToAtlasTexelIdx(src_probe_idx, src_line_idx.xy, src_probe_points_count);

                color += texelFetch(src_cascade_atlas, src_atlas_texel_idx, 0).rgba * weights[sample_idx] * (step_size * step_size);
              }
            }
          }
        }
      }
    }
  }

  //if(dst_interval_idx.line_idx.y == 6 && dst_interval_idx.line_idx.x == 3) color += vec4(0.0f, 1.0f, 0.0f, 0.0f);


  /*if(dst_interval_idx.block_idx.x == 0 && dst_interval_idx.block_idx.y == 0 && dst_interval_idx.line_idx.x == 1 && dst_interval_idx.line_idx.y == 13)
  {
    color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
  }*/
  //if(dst_interval_idx.line_idx.x == 3 && dst_interval_idx.line_idx.y == 6)
  /*{
    for(uint y_offset = 0u; y_offset < 2u; y_offset++)
    {
      for(uint x_offset = 0u; x_offset < 2u; x_offset++)
      {
        //ivec2 src_probe_idx = ivec2(4, 3);
        ivec2 src_probe_idx = dst_interval_idx.block_idx * 2 + ivec2(x_offset, y_offset);
        
        uint count = 10u;
        for(uint y = 0u; y < count; y++)
        {
          for(uint x = 0u; x < count; x++)
          {
            vec2 ratio = (vec2(x, y) + vec2(0.5f)) / float(count);
            RaySample ray_sample = SampleBundleRay(dst_edge0, dst_edge1, ratio);
            ProbeHit src_probe_hit = RayToPointIdsf(src_probe_idx, ray_sample.ray_start, normalize(ray_sample.ray_end - ray_sample.ray_start), src_probe_points_count, src_probe_spacing);
            if(src_probe_hit.is_hit)
            {
              ivec2 src_line_idx = ivec2(floor(src_probe_hit.line_idxf));
              ivec2 src_atlas_texel_idx = IntervalIdxToAtlasTexelIdx(src_probe_idx, src_line_idx, src_probe_points_count);

              //color += texelFetch(src_cascade_atlas, src_atlas_texel_idx, 0).rgba * ray_sample.weight / float(count * count);// * (src_probe_hit.t.y - src_probe_hit.t.x) / float(count * count);
            }
          }
        }
      }
    }
  }*/
  //color /= color.a + 1e-7f;
  //if(dst_interval_idx.line_idx.y == 6) color = vec4(1.0f, 0.5f, 0.0f, 1.0f);
}}

[include: "pcg", "block_probe_layout2", "hrc_basis"]
void FinalGatheringShader(
  uint c0_probe_spacing,
  uint c0_probe_points_count,
  uint cascade_idx,
  sampler2D cascade_atlas,
  out vec4 color)
{{
  vec2 pixel_pos = gl_FragCoord.xy;
  uint probe_spacing = GetProbeSpacing(c0_probe_spacing, cascade_idx);
  uint block_lines_count2 = GetProbePointsCount(c0_probe_points_count, cascade_idx);
  vec2 probe_idxf = pixel_pos / vec2(probe_spacing);
  ivec2 block_idx = ivec2(floor(probe_idxf));
  //ivec2 block_idx = ivec2(7, 5) / int(1u << cascade_idx);

  vec4 pixel_aabb = vec4(pixel_pos - vec2(0.5f), pixel_pos + vec2(0.5f));

  /*color = vec4(0.0f);
  for(uint point_idx0 = 0u; point_idx0 < block_lines_count2 * 4u; point_idx0++)
  {
    for(uint point_idx1 = 0u; point_idx1 < block_lines_count2 * 4u; point_idx1++)
    {
      ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(block_idx, ivec2(point_idx0, point_idx1), block_lines_count2);
      vec4 edge0 = GetProbePointEdge(block_idx, int(point_idx0), block_lines_count2, probe_spacing);
      vec4 edge1 = GetProbePointEdge(block_idx, int(point_idx1), block_lines_count2, probe_spacing);

      vec2 midpoint0 = mix(edge0.xy, edge0.zw, 0.5f);
      vec2 midpoint1 = mix(edge1.xy, edge1.zw, 0.5f);
      vec2 t;
      bool aabb_hit = RayAABBIntersect(pixel_aabb, midpoint0, normalize(midpoint1 - midpoint0), t.x, t.y);
      if(aabb_hit)
      {
        color += texelFetch(cascade_atlas, atlas_texel_idx, 0) * (t.y - t.x);
      }
    }
  }*/

  color = vec4(0.0f);
  uvec2 line_idx;
  for(line_idx.x = 0u; line_idx.x < block_lines_count2; line_idx.x++)
  //line_idx.x = 0u;
  {
    //vec4 base_edge = GetProbePointEdge(block_idx, int(point_idx0), block_lines_count2, probe_spacing);
    //vec2 base_midpoint = mix(base_edge.xy, base_edge.zw, 0.5f);
    //ProbeHit src_probe_hit = RayToPointIdsf(block_idx, base_midpoint, normalize(pixel_pos - base_midpoint), block_lines_count2, probe_spacing);
    //for(line_idx.y = 0u; line_idx.y < block_lines_count2; line_idx.y++)
    float line_idx_yf = FindLineIdxY(block_idx, float(line_idx.x), pixel_pos, block_lines_count2, probe_spacing);
    line_idx.y = uint(round(line_idx_yf));
    {
      //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
      //color += vec4(0.1f);
      {
        //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
        //Ray ray = LineIdxfToRay(block_idx, vec2(line_idx.x, line_idx_yf), block_lines_count2, probe_spacing);

        ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(block_idx, ivec2(line_idx), block_lines_count2);

        //ivec2 test_probe_idx = ivec2(8, 17);
        //vec4 test_aabb = vec4(test_probe_idx, test_probe_idx + ivec2(1)) * float(c0_probe_spacing);

        /*vec2 t;
        bool pixel_aabb_hit = RayAABBIntersect(pixel_aabb, ray.origin, normalize(ray.dir), t.x, t.y);
        if(pixel_aabb_hit && light_aabb_hit)
        {
          color += vec4(1.0f, 0.5f, 0.0f, 1.0f) * 5e-3;
        }*/

        vec2 t;
        //bool aabb_hit = RayAABBIntersect(pixel_aabb, midpoint0, normalize(midpoint1 - midpoint0), t.x, t.y);
        //if(aabb_hit)
        {
          color += texelFetch(cascade_atlas, atlas_texel_idx, 0);// * (t.y - t.x);
        }
      }
    }
  }

  /*uint count = 100u;
  for(uint dir_idx = 0u; dir_idx < count; dir_idx++)
  {
    float ratio = (float(dir_idx) + 0.5f) / float(count);
    float ang = ratio * pi * 2.0f;
    vec2 ray_dir = vec2(cos(ang), sin(ang));
    vec2 point_idxf = RayToPointIdsf(block_idx, pixel_pos, ray_dir, block_lines_count2, probe_spacing);
    if(point_idxf.x > 3.0f && point_idxf.x < 3.99f && point_idxf.y > 0.0f && point_idxf.y < 1.0f)
    {
      color += vec4(1.0f, 0.5f, 0.0f, 0.0f) / float(count);
    }
  }*/



  //ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(block_idx, ivec2(1, 3), block_lines_count2);
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
  uint block_lines_count2 = GetProbePointsCount(c0_probe_points_count, cascade_idx);
  uvec2 probe_spacing = uvec2(GetProbeSpacing(c0_probe_spacing, cascade_idx));

  IntervalIdx interval_idx = AtlasTexelIdxToIntervalIdx(atlas_texel_idx, block_lines_count2);
  atlas_texel = vec4(0.0f, 0.0f, 0.0f, 1.0f);

  if((interval_idx.block_idx.x == 32) && (interval_idx.block_idx.y == 47))
  //if(length(vec2(interval_idx.block_idx) - vec2(160.0f, 50.0f)) < 2.0f)
  {
    atlas_texel = vec4(1.0f, 0.5f, 0.0f, 0.0f) * 0.1f;
  }

  /*if(interval_idx.block_idx.x == 0 && interval_idx.block_idx.y == 0 && interval_idx.line_idx.x == 0 && interval_idx.line_idx.y == 5)
  {
    atlas_texel = vec4(0.0f, 1.0f, 0.0f, 1.0f);
  }else
  {
    atlas_texel = vec4(0.0f);
  }*/
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

  const int cascades_count = 9;

  array<Image> extended_cascades;
  array<Image> merged_cascades;

  uint c0_probe_spacing = 3;
  uint c0_probe_points_count = 1;

  uint curr_probe_spacing = c0_probe_spacing;
  uint curr_probe_points_count = c0_probe_points_count;

  uvec2 c0_probes_count = viewport_size / c0_probe_spacing;
  for(uint cascade_idx = 0; cascade_idx < cascades_count; cascade_idx++)
  {
    uvec2 curr_probes_count = (viewport_size + uvec2(curr_probe_spacing - 1)) / curr_probe_spacing;
    uvec2 curr_size = curr_probes_count * curr_probe_points_count;
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

  for(uint src_cascade_idx = 0; src_cascade_idx < cascades_count - 1; src_cascade_idx++)
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
  int gather_cascade_idx = SliderInt("Gather cascade_idx", 0, cascades_count - 1, 8);

  FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    gather_cascade_idx,
    extended_cascades[gather_cascade_idx],
    GetSwapchainImage()
  );
  /*CopyShader(
    extended_cascades[7],
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


[declaration: "block_probe_layout"]
[include: "geometry_utils"]
{{
  struct IntervalIdx
  {
    ivec2 block_idx;
    ivec2 line_idx;
  };

  IntervalIdx AtlasTexelIdxToIntervalIdx(ivec2 atlas_texel_idx, uint block_lines_count2)
  {
    uvec2 block_size = uvec2(block_lines_count2);
    IntervalIdx interval_idx;
    interval_idx.block_idx = atlas_texel_idx / int(block_size);
    interval_idx.line_idx = atlas_texel_idx % int(block_size);
    return interval_idx;
  }

  ivec2 IntervalIdxToAtlasTexelIdx(ivec2 block_idx, ivec2 line_idx, uint block_lines_count2)
  {
    uvec2 block_size = uvec2(block_lines_count2);
    return ivec2(block_size) * block_idx + line_idx;
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
    float eps = 1e-2f;
    if(point.y < eps)
      return (point.x + 0.0f) / 4.0f;
    if(point.x > 1.0f - eps)
      return (point.y + 1.0f) / 4.0f;
    if(point.y > 1.0f - eps)
      return ((1.0f - point.x) + 2.0f) / 4.0f;
    return ((1.0f - point.y) + 3.0f) / 4.0f;
  }


  const vec2 norm_circle_center = vec2(0.5f, 0.5f);
  const float norm_circle_radius = sqrt(2.0f) / 2.0f;

  vec2 GetNormCirclePoint(float ratio)
  {
    float ang = fract(ratio) * 2.0f * 3.141592f;
    return norm_circle_center + vec2(cos(ang), sin(ang)) * norm_circle_radius;
  }

  float GetNormCircleRatio(vec2 circle_point)
  {
    vec2 delta = circle_point - norm_circle_center;
    float ang = atan(delta.y, delta.x);
    return fract(ang / (2.0f * 3.141592f));
  }

  vec2 RealToLinespaceAABB(vec4 norm_edge)
  {
    vec2 norm_dir = norm_edge.zw - norm_edge.xy;
    vec2 t;
    bool is_hit = RayAABBIntersect(vec4(0.0f, 0.0f, 1.0f, 1.0f), norm_edge.xy, norm_dir, t.x, t.y);
    vec4 norm_inter_edge;
    norm_inter_edge.xy = norm_edge.xy + norm_dir * t.x;
    norm_inter_edge.zw = norm_edge.xy + norm_dir * t.y;
    vec2 linespace = vec2(GetNormAAABBPerimeterRatio(norm_inter_edge.xy), GetNormAAABBPerimeterRatio(norm_inter_edge.zw));
    if(!is_hit)
      linespace = vec2(-1.0f);
    return linespace;
  }
  vec4 LinespaceToRealAABB(vec2 line_coords)
  {
    return vec4(GetNormAABBPerimeterPoint(line_coords.x), GetNormAABBPerimeterPoint(line_coords.y));
    //return vec4(GetNormCirclePoint(line_coords.x), GetNormCirclePoint(line_coords.y));
  }

  vec2 RealToLinespaceCircle(vec4 norm_edge)
  {
    vec2 norm_dir = norm_edge.zw - norm_edge.xy;
    vec2 t = RaySphereIntersect(norm_edge.xy, norm_dir, norm_circle_center, norm_circle_radius);
    bool is_hit = t.x < 1e5f;

    vec4 norm_inter_edge;
    norm_inter_edge.xy = norm_edge.xy + norm_dir * t.x;
    norm_inter_edge.zw = norm_edge.xy + norm_dir * t.y;
    vec2 linespace = vec2(GetNormCircleRatio(norm_inter_edge.xy), GetNormCircleRatio(norm_inter_edge.zw));
    if(!is_hit)
      linespace = vec2(-1.0f);
    return linespace;
  }

  vec4 LinespaceToRealCircle(vec2 line_coords)
  {
    //return vec4(GetNormAABBPerimeterPoint(line_coords.x), GetNormAABBPerimeterPoint(line_coords.y));
    return vec4(GetNormCirclePoint(line_coords.x), GetNormCirclePoint(line_coords.y));
  }

  vec2 RealToLinespaceAngdist(vec4 norm_edge)
  {
    vec2 norm_dir = norm_edge.zw - norm_edge.xy;
    vec2 norm_perp = normalize(vec2(-norm_dir.y, norm_dir.x));

    float ang = atan(norm_dir.y, norm_dir.x);
    return vec2(fract(ang / (pi * 2.0f)), dot(norm_perp, norm_edge.xy - vec2(0.5f)) + 0.5f);
  }

  vec4 LinespaceToRealAngdist(vec2 line_coords)
  {
    float ang = line_coords.x * pi * 2.0f;
    vec2 dir = vec2(cos(ang), sin(ang));
    vec2 perp = vec2(-dir.y, dir.x);
    vec2 origin = vec2(0.5f) + perp * (line_coords.y - 0.5f);
    return vec4(origin - dir * 0.0f, origin + dir * 1.0f);
  }

  //#define LinespaceToReal LinespaceToRealAABB
  //#define RealToLinespace RealToLinespaceAABB
  //#define LinespaceToReal LinespaceToRealCircle
  //#define RealToLinespace RealToLinespaceCircle
  #define LinespaceToReal LinespaceToRealAngdist
  #define RealToLinespace RealToLinespaceAngdist

  vec4 IntervalIdxToIntervalPoints(ivec2 block_idx, ivec2 line_idx, uint block_lines_count2, uint probe_spacing)
  {
    vec2 ratio = (vec2(line_idx) + vec2(0.5f)) / float(block_lines_count2);
    vec4 norm_points = LinespaceToReal(ratio);
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    return vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_points.zw
    );
  }

  vec4 GetProbePointEdge(ivec2 block_idx, int point_idx, uint block_lines_count2, uint probe_spacing)
  {
    vec2 edge_ratio = vec2(point_idx, point_idx + 1) / float(block_lines_count2);

    vec4 norm_edge_points = LinespaceToReal(edge_ratio);

    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    return vec4(
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge_points.xy,
      probe_minmax.xy + (probe_minmax.zw - probe_minmax.xy) * norm_edge_points.zw
    );
  }

  struct ProbeHit
  {
    bool is_hit;
    vec2 line_idxf;
  };
  ProbeHit RayToPointIdsf(ivec2 block_idx, vec2 ray_origin, vec2 ray_dir, uint block_lines_count2, uint probe_spacing)
  {
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    vec2 norm_ray_origin = (ray_origin - probe_minmax.xy) / float(probe_spacing);
    vec2 norm_ray_end = (ray_origin + ray_dir - probe_minmax.xy) / float(probe_spacing);

    ProbeHit probe_hit;
    vec2 norm_linespace = RealToLinespace(vec4(norm_ray_origin, norm_ray_end));
    probe_hit.line_idxf = norm_linespace * float(block_lines_count2) - vec2(0.5f);
    probe_hit.is_hit = norm_linespace.x >= 0.0f;

    return probe_hit;
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

