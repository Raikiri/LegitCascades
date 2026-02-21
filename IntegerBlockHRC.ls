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

  vec2 RayRayIntersect(vec2 origin0, vec2 dir0, vec2 origin1, vec2 dir1)
  {
    //origin0 + dir0 * t0 == origin1 + dir1 * t1
    //dir0 * t0 - dir1 * t1 = origin1 - origin0
    mat2 m = mat2(dot(dir0, dir0), dot(dir1, dir0), -dot(dir0, dir1), -dot(dir1, dir1));
    vec2 rhs = vec2(dot(origin1 - origin0, dir0), dot(origin1 - origin0, dir1));
    return inverse(m) * rhs;
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
  struct Ray
  {
    vec2 origin;
    vec2 dir;
  };
}}

[declaration: "pixel_integer_lines"]
{{
  int Sign(int v)
  {
    return v < 0 ? -1 : 1;
  }
  ivec2 MinRatio(ivec2 r0, ivec2 r1)
  {
    //r0.x / r0.y < r1.x / r1.y ? r0 : r1
    ivec2 u0 = r0 * Sign(r0.y);
    ivec2 u1 = r1 * Sign(r0.y);
    return (u0.x * u1.y < u1.x * u0.y) ? r0 : r1;
  }

  ivec2 MaxRatio(ivec2 r0, ivec2 r1)
  {
    ivec2 u0 = r0 * Sign(r0.y);
    ivec2 u1 = r1 * Sign(r0.y);
    return (u0.x * u1.y > u1.x * u0.y) ? r0 : r1;
  }

  int IntegerLineLen(ivec4 line_points)
  {
    ivec2 delta = line_points.zw - line_points.xy;
    return max(abs(delta.x), abs(delta.y));
  }

  struct LineSegments2
  {
    ivec4 segment_points[2];
    uint mask;
  };

  int FloorDiv(int a, int b)
  {
    int d = b * 100000;
    return (a + d) / b - d / b;
  }

  int RoundDiv(int a, int b)
  {
    /*if(a < 0)
    {
      a *= -1;
      b *= -1;
    }*/
    
    int d = b * 100000;
    //return (a + (b >> 1) + d) / b - d / b;
    return (a + (b - 1) / 2 + d) / b - d / b;
  }

  ivec2 RoundDiv(ivec2 a, int b)
  {
    return ivec2(RoundDiv(a.x, b), RoundDiv(a.y, b));
  }

  ivec2 GetLinePoint(ivec4 line_points, ivec2 ratio)
  {
    ratio *= Sign(ratio.y);
    return line_points.xy + RoundDiv((line_points.zw - line_points.xy) * ratio.x, ratio.y);
  }

  bool GreaterOrEq(ivec2 ratio, int v)
  {
    ratio *= Sign(ratio.y);
    return ratio.x >= ratio.y * v;
  }
  bool LesserOrEq(ivec2 ratio, int v)
  {
    ratio *= Sign(ratio.y);
    return ratio.x <= ratio.y * v;
  }
  LineSegments2 SplitLineX(ivec4 line_points, int x)
  {
    if(line_points.x > line_points.z)
    {
      ivec2 t = line_points.xy;
      line_points.xy = line_points.zw;
      line_points.zw = t;
    }

    LineSegments2 segments;
    segments.mask = 0u;
    if(line_points.x == line_points.z)
    {
      if(line_points.x < x)
      {
        segments.segment_points[0] = line_points;
        segments.segment_points[1] = ivec4(0);
        segments.mask = 1u;
        return segments;
      }else
      {
        segments.segment_points[0] = ivec4(0);
        segments.segment_points[1] = line_points;
        segments.mask = 2u;
        return segments;
      }
    }else
    {
      bool dir = line_points.x <= line_points.z;
      ivec2 ratio_left = ivec2(x - (dir ? 1 : 0) - line_points.x, line_points.z - line_points.x);
      ivec2 ratio_right = ivec2(x - (!dir ? 1 : 0) - line_points.x, line_points.z - line_points.x);

      segments.segment_points[0] = ivec4(line_points.xy, GetLinePoint(line_points, ratio_left));
      segments.segment_points[1] = ivec4(GetLinePoint(line_points, ratio_right), line_points.zw);
      if(GreaterOrEq(ratio_left, 0) && LesserOrEq(ratio_left, 1)) segments.mask |= 1u;
      if(GreaterOrEq(ratio_right, 0) && LesserOrEq(ratio_right, 1)) segments.mask |= (1u << 1u);
    }
    return segments;
  }

  LineSegments2 SplitLineY(ivec4 line_points, int y)
  {
    LineSegments2 segments = SplitLineX(line_points.yxwz, y);
    segments.segment_points[0] = segments.segment_points[0].yxwz;
    segments.segment_points[1] = segments.segment_points[1].yxwz;
    return segments;
  }

  ivec2 IntegerLineVsAABB(ivec4 line_points, ivec4 aabb_minmax)
  {
    ivec2 delta = line_points.zw - line_points.xy;
    ivec2 p = line_points.xy;

    ivec2 x1_ratio = ivec2(aabb_minmax.x - p.x, delta.x);
    ivec2 y1_ratio = ivec2(aabb_minmax.y - p.y, delta.y);

    ivec2 x2_ratio = ivec2(aabb_minmax.z - p.x, delta.x);
    ivec2 y2_ratio = ivec2(aabb_minmax.w - p.y, delta.y);

    ivec2 min_ratio = MaxRatio(MinRatio(x1_ratio, x2_ratio), MinRatio(y1_ratio, y2_ratio));
    ivec2 max_ratio = MinRatio(MaxRatio(x1_ratio, x2_ratio), MaxRatio(y1_ratio, y2_ratio));

    int line_len = IntegerLineLen(line_points);
    //min_ratio *= Sign(min_ratio.y);
    //max_ratio *= Sign(max_ratio.y);
    return ivec2(min_ratio.y == 0 ? 0x0 : min_ratio.x * line_len / min_ratio.y, max_ratio.y == 0 ? 0xffff : max_ratio.x * line_len / max_ratio.y);
  }

  bool IsPointInInterval(int p, ivec2 interval)
  {
    return (p >= interval.x && p <= interval.y) || (p >= interval.y && p <= interval.x);
  }
  int IsPointOnLine(ivec2 point, ivec4 line_points)
  {
    if(!IsPointInInterval(point.x, line_points.xz)) return -1;
    ivec4 curr_line_points = line_points;
    for(int i = 0; i < 10; i++)
    {
      if(abs(curr_line_points.z - curr_line_points.x) == 0) return IsPointInInterval(point.y, curr_line_points.yw) ? i : -6;
      int x_midpoint = (curr_line_points.x + curr_line_points.z) / 2 + 1;
      //int x_midpoint = FloorDiv(curr_line_points.x + curr_line_points.z, 2) + 1;
      LineSegments2 segments = SplitLineX(curr_line_points, x_midpoint);
      if(IsPointInInterval(point.x, ivec2(segments.segment_points[0].x, segments.segment_points[0].z)))
      {
        if((segments.mask & 1u) == 0u) return -2;
        curr_line_points = segments.segment_points[0];
      }else
      if(IsPointInInterval(point.x, ivec2(segments.segment_points[1].x, segments.segment_points[1].z)))
      {
        if((segments.mask & 2u) == 0u) return -3;
        curr_line_points = segments.segment_points[1];
      }else
      {
        return -4;
      }
    }
    return -5;
  }

  bool IsPointInAabb(ivec2 point, ivec4 aabb_minmax)
  {
    return point.x >= aabb_minmax.x && point.x <= aabb_minmax.z && point.y >= aabb_minmax.y && point.y <= aabb_minmax.w;
  }

  ivec4 GetChildAabb(ivec4 aabb, ivec2 split, int child_idx)
  {
    ivec4 child_aabb;
    if(child_idx == 0 || child_idx == 2)
    {
      child_aabb.x = aabb.x;
      child_aabb.z = split.x - 1;
    }else
    {
      child_aabb.x = split.x;
      child_aabb.z = aabb.z;
    }
    if(child_idx == 0 || child_idx == 1)
    {
      child_aabb.y = aabb.y;
      child_aabb.w = split.y - 1;
    }else
    {
      child_aabb.y = split.y;
      child_aabb.w = aabb.w;
    }
    return child_aabb;
  }
  struct LineSegments4
  {
    ivec4 line_points[4];
    uint mask;
  };
  /*LineSegments4 SplitLineXY(ivec4 line, ivec2 split)
  {
    LineSegments2 y_segments = SplitLineY(line, split.y);
    LineSegments2 x_segments0 = SplitLineX(y_segments.segment_points[0], split.x);
    LineSegments2 x_segments1 = SplitLineX(y_segments.segment_points[1], split.x);

    LineSegments4 segments;
    segments.mask = 0u;
    segments.line_points[0] = x_segments0.segment_points[0];
    segments.segments.mask = (x_segments0.mask & 1) == 1) &&
    
  }*/

  struct LineSegment
  {
    ivec4 segment_points;
    bool is_present;
  };
  LineSegment GetChildLine(ivec4 line, ivec2 split, int child_idx)
  {
    LineSegment segment;
    LineSegments2 y_segments = SplitLineY(line, split.y);
    if(child_idx == 0 || child_idx == 1)
    {
      LineSegments2 x_segments = SplitLineX(y_segments.segment_points[0], split.x);
      if(child_idx == 0)
      {
        segment.segment_points = x_segments.segment_points[0];
        segment.is_present = ((x_segments.mask & 1u) == 1u) && ((y_segments.mask & 1u) == 1u);
        return segment;
      }else
      {
        segment.segment_points = x_segments.segment_points[1];
        segment.is_present = (((x_segments.mask >> 1u) & 1u) == 1u) && ((y_segments.mask & 1u) == 1u);
        return segment;
      }
    }else
    {
      LineSegments2 x_segments = SplitLineX(y_segments.segment_points[1], split.x);
      if(child_idx == 2)
      {
        segment.segment_points = x_segments.segment_points[0];
        segment.is_present = ((x_segments.mask & 1u) == 1u) && (((y_segments.mask >> 1u) & 1u) == 1u);
        return segment;
      }else
      {
        segment.segment_points = x_segments.segment_points[1];
        segment.is_present = (((x_segments.mask >> 1u) & 1u) == 1u) && (((y_segments.mask >> 1u) & 1u) == 1u);
        return segment;
      }
    }
  }
  int GetChildIdx(ivec2 point, ivec2 split)
  {
    int idx = 0;
    if(point.x >= split.x) idx += 1;
    if(point.y >= split.y) idx += 2;
    return idx;
  }
  int IsPointOnAabbLine(ivec2 point, ivec4 line_points, ivec4 aabb_minmax)
  {
    ivec4 curr_aabb = aabb_minmax;
    if(!IsPointInAabb(point, curr_aabb))
      return -1;
    ivec4 curr_line_points = line_points;

    for(int i = 0; i < 10; i++)
    {
      if(!IsPointInAabb(curr_line_points.xy, curr_aabb)) return -4;
      if(!IsPointInAabb(curr_line_points.zw, curr_aabb)) return -5;
      if(curr_aabb.x == curr_aabb.z) return i;

      ivec2 split;
      split.x = (curr_aabb.x + curr_aabb.z) / 2 + 1;
      split.y = (curr_aabb.y + curr_aabb.w) / 2 + 1;

      int child_idx = GetChildIdx(point, split);
      curr_aabb = GetChildAabb(curr_aabb, split, child_idx);
      LineSegment child_segment = GetChildLine(curr_line_points, split, child_idx);
      if(!child_segment.is_present) return -2;
      curr_line_points = child_segment.segment_points;
    }
    return -3;
  }
}}

[include: "pcg", "pixel_integer_lines"]
void IntegerTestShader(
  out vec4 color)
{{
  ivec2 pixel_idx = ivec2(gl_FragCoord.xy);
  ivec2 tile_idx = pixel_idx / 30;
  float checkerboard = (tile_idx.x + tile_idx.y) % 2 == 0 ? 1.0f : 0.0f;
  color = vec4(0.4f + 0.1f * checkerboard);

  ivec4 test_aabb_minmax = ivec4(0, 0, 3, 3);
  ivec4 test_line_points = ivec4(3, 0, 0, 3);
  //ivec4 test_line_points = ivec4(27, 15, 0, 2);

  //int res = IsPointOnLine(tile_idx, test_line_points);
  int res = IsPointOnAabbLine(tile_idx, test_line_points, test_aabb_minmax);
  if(res >= 0)
  {
    color += vec4(0.0f, 0.3f * float(res + 1), 0.0f, 0.0f);
  }
  /*if(res == -1)
    color += vec4(0.0f, 0.0f, 1.0f, 1.0f);*/
  if(res == -2)
    color += vec4(1.0f, 1.0f, 0.0f, 1.0f);
  if(res == -3)
    color += vec4(0.0f, 0.0f, 1.0f, 1.0f);
  if(res == -4)
    color += vec4(1.0f, 0.0f, 1.0f, 1.0f);
  if(res == -5)
    color += vec4(1.0f, 0.0f, 1.0f, 1.0f);
  if(res == -6)
    color += vec4(1.0f, 1.0f, 0.0f, 1.0f);

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

[declaration: "circle_linespace"]
[include: "geometry_utils"]
{{
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

  //try torus of lines?
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

  float GetLineWeight(vec2 line_coord)
  {
    return 1.0f;
  }
}}


[declaration: "diagonal_linespace"]
[include: "geometry_utils"]
{{
  Ray LineCoordToRay(vec2 line_coord)
  {
    float ang = line_coord.x * 2.0f * pi;
    
    vec4 edge0 = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    vec4 edge1 = vec4(1.0f, 0.0f, 0.0f, 1.0f);

    Ray ray;
    ray.dir = vec2(cos(ang), sin(ang));
    if(abs(dot(ray.dir, edge0.zw - edge0.xy)) < abs(dot(ray.dir, edge1.zw - edge1.xy)))
    {
      ray.origin = mix(edge0.xy, edge0.zw, line_coord.y);
    }else
    {
      ray.origin = mix(edge1.xy, edge1.zw, line_coord.y);
    }
    
    return ray;
  }

  vec2 RayToLineCoord(vec2 origin, vec2 dir)
  {
    float ang = atan(dir.y, dir.x);
    vec2 line_coord;
    line_coord.x = fract(ang / (2.0f * pi));

    vec4 edge0 = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    vec4 edge1 = vec4(1.0f, 0.0f, 0.0f, 1.0f);

    vec4 edge;
    if(abs(dot(dir, edge0.zw - edge0.xy)) < abs(dot(dir, edge1.zw - edge1.xy)))
    {
      edge = edge0;
    }else
    {
      edge = edge1;
    }

    vec2 params = RayRayIntersect(origin, dir, edge.xy, edge.zw - edge.xy);

    line_coord.y = params.y;
    return line_coord;
  }

  float FindLineCoordY(vec2 norm_ray_point, float line_coord_x)
  {
    float ang = line_coord_x * 2.0f * pi;

    Ray ray;
    ray.origin = norm_ray_point;
    ray.dir = vec2(cos(ang), sin(ang));

    vec4 edge0 = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    vec4 edge1 = vec4(1.0f, 0.0f, 0.0f, 1.0f);

    vec4 edge;
    if(abs(dot(ray.dir, edge0.zw - edge0.xy)) < abs(dot(ray.dir, edge1.zw - edge1.xy)))
    {
      edge = edge0;
    }else
    {
      edge = edge1;
    }

    vec2 params = RayRayIntersect(ray.origin, ray.dir, edge.xy, edge.zw - edge.xy);
    return params.y;
  }

  float GetLineWeight(vec2 line_coord)
  {
    float ang = line_coord.x * 2.0f * pi;
    vec2 dir = vec2(cos(ang), sin(ang));

    vec2 perp = vec2(-dir.y, dir.x);
    vec4 edge0 = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    vec4 edge1 = vec4(1.0f, 0.0f, 0.0f, 1.0f);

    if(abs(dot(dir, edge0.zw - edge0.xy)) < abs(dot(dir, edge1.zw - edge1.xy)))    
    {
      return 1.0f / abs(dot(perp, normalize(edge0.zw - edge0.xy)));
    }else
    {
      return 1.0f / abs(dot(perp, normalize(edge1.zw - edge1.xy)));
    }
  }
}}


[declaration: "line_delta_linespace"]
[include: "geometry_utils"]
{{
  Ray LineCoordToRay(vec2 line_coord)
  {
    vec2 p0 = vec2(-1.0f + line_coord.x * 3.0f, 0.0f);
    vec2 p1 = vec2(line_coord.x + (line_coord.y - 0.5f) * 2.0f, 1.0f);

    Ray ray;
    ray.origin = p0;
    ray.dir = p1 - p0;
    
    return ray;
  }

  vec2 RayToLineCoord(vec2 origin, vec2 dir)
  {
    vec2 t;
    vec2 params0 = RayRayIntersect(origin, dir, vec2(0.0f, 0.0f), vec2(1.0f, 0.0f));
    vec2 params1 = RayRayIntersect(origin, dir, vec2(0.0f, 1.0f), vec2(1.0f, 0.0f));

    vec2 p0 = origin + dir * params0.x;
    vec2 p1 = origin + dir * params1.x;

    vec2 line_coord;
    line_coord.x = (p0.x + 1.0f) / 3.0f;
    line_coord.y = (p1.x - line_coord.x) / 2.0f + 0.5f;
    return line_coord;
  }

  float FindLineCoordY(vec2 norm_ray_point, float line_coord_x)
  {
    vec2 p0 = vec2(-1.0f + line_coord_x * 3.0f, 0.0f);
    vec2 params = RayRayIntersect(p0, norm_ray_point - p0, vec2(0.0f, 1.0f), vec2(1.0f, 0.0f));
    vec2 p1 = vec2(params.y, 1.0f);
    return (p1.x - line_coord_x) / 2.0f + 0.5f;
  }

  float GetLineWeight(vec2 line_coord)
  {
    return 1.0f;
  }
}}

[declaration: "block_probe_layout2"]
[include: "diagonal_linespace"]
{{
  #define TEXEL_CENTER 0.5f
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
  float GetLineIdxfWeight(ivec2 block_idx, vec2 line_idxf, uint block_lines_count2, uint probe_spacing)
  {
    vec2 line_coord = (line_idxf + vec2(TEXEL_CENTER)) / float(block_lines_count2);
    return GetLineWeight(line_coord);
  }
  Ray LineIdxfToRay(ivec2 block_idx, vec2 line_idxf, uint block_lines_count2, uint probe_spacing)
  {
    vec2 line_coord = (line_idxf + vec2(TEXEL_CENTER)) / float(block_lines_count2);
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
      return line_coord * float(block_lines_count2) - vec2(TEXEL_CENTER);
    else
      return vec2(1e7f);
  }

  float FindLineIdxY(ivec2 block_idx, float line_idx_x, vec2 ray_point, uint block_lines_count2, uint probe_spacing)
  {
    vec4 probe_minmax = vec4(vec2(block_idx), vec2(block_idx + ivec2(1))) * float(probe_spacing);
    vec2 norm_ray_point = (ray_point - probe_minmax.xy) / float(probe_spacing);
    return FindLineCoordY(norm_ray_point, (line_idx_x + TEXEL_CENTER) / float(block_lines_count2)) * float(block_lines_count2) - TEXEL_CENTER;
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
  float total_weight = 0.0f;

  vec2 ratio;
  uint steps_count = 1u;
  uvec2 step_idx;
  for(step_idx.y = 0u; step_idx.y < steps_count; step_idx.y++)
  {
    for(step_idx.x = 0u; step_idx.x < steps_count; step_idx.x++)
    {
      vec2 ratio = (vec2(step_idx) + vec2(0.5f)) / float(steps_count);
      //float weight = 1.0f;//GetLineIdxfWeight(dst_interval_idx.block_idx, dst_line_idx, dst_probe_points_count, dst_probe_spacing);
      //total_weight += weight;
      vec2 dst_line_idx = vec2(dst_interval_idx.line_idx) - vec2(0.5f) + ratio;
      Ray dst_ray = LineIdxfToRay(dst_interval_idx.block_idx, dst_line_idx, dst_probe_points_count, dst_probe_spacing);
      for(uint y_offset = 0u; y_offset < 2u; y_offset++)
      {
        for(uint x_offset = 0u; x_offset < 2u; x_offset++)
        {
          //ivec2 src_probe_idx = ivec2(4, 3);
          ivec2 src_probe_idx = dst_interval_idx.block_idx * 2 + ivec2(x_offset, y_offset);  
          vec2 src_line_idx = RayToLineIdxf(src_probe_idx, dst_ray.origin, dst_ray.dir, src_probe_points_count, src_probe_spacing);

          //float weight = GetLineIdxfWeight(src_probe_idx, src_line_idx, src_probe_points_count, src_probe_spacing);
          //total_weight += weight / 4.0f;

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
            vec4 probe_total_radiance = vec4(0.0f);
            float probe_total_weight = 0.0f;

            BilinearSamples bilinear_samples = GetBilinearSamples(src_line_idx);
            vec4 weights = GetBilinearWeights(round(bilinear_samples.ratio));
            for(uint sample_idx = 0u; sample_idx < 4u; sample_idx++)
            {
              //ivec2 src_line_idx = (bilinear_samples.base_idx + GetBilinearOffset(sample_idx) + ivec2(src_probe_points_count)) % ivec2(src_probe_points_count);
              ivec2 src_line_idx = bilinear_samples.base_idx + GetBilinearOffset(sample_idx);
              if(src_line_idx.x >= 0 && src_line_idx.y >= 0 && src_line_idx.x < int(src_probe_points_count) && src_line_idx.y < int(src_probe_points_count))
              {
                float weight = weights[sample_idx];// * GetLineIdxfWeight(src_probe_idx, vec2(src_line_idx), src_probe_points_count, src_probe_spacing);
                probe_total_weight += weight;

                ivec2 src_atlas_texel_idx = IntervalIdxToAtlasTexelIdx(src_probe_idx, src_line_idx.xy, src_probe_points_count);
                probe_total_radiance += texelFetch(src_cascade_atlas, src_atlas_texel_idx, 0).rgba * weight;
              }
            }

            //probe_total_radiance /= probe_total_weight;
            color += probe_total_radiance;// * weight;
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
    int t = int(round(line_idx_yf));
    if(t >= 0 && t < int(block_lines_count2))
    {
      //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
      //color += vec4(0.1f);
      {
        //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
        //float line_idx_yf = float(line_idx.y);
        Ray discrete_ray = LineIdxfToRay(block_idx, vec2(line_idx.x, line_idx.y), block_lines_count2, probe_spacing);
        Ray continuous_ray = LineIdxfToRay(block_idx, vec2(line_idx.x, line_idx_yf), block_lines_count2, probe_spacing);

        ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(block_idx, ivec2(line_idx), block_lines_count2);

        ivec2 test_probe_idx = ivec2(3, 7);
        vec4 test_aabb = vec4(test_probe_idx, test_probe_idx + ivec2(1)) * float(c0_probe_spacing);

        float weight = GetLineIdxfWeight(block_idx, vec2(float(line_idx.x), line_idx_yf), block_lines_count2, probe_spacing);
        
        /*vec2 pixel_t;
        bool pixel_aabb_hit = RayAABBIntersect(pixel_aabb, discrete_ray.origin, normalize(discrete_ray.dir), pixel_t.x, pixel_t.y);
        vec2 light_t;
        bool light_aabb_hit = RayAABBIntersect(test_aabb, discrete_ray.origin, normalize(discrete_ray.dir), light_t.x, light_t.y);
        if(pixel_aabb_hit && light_aabb_hit)
        {
          vec4 mult = vec4(1.0f);
          vec2 rec_line_idx = RayToLineIdxf(block_idx, discrete_ray.origin, discrete_ray.dir, block_lines_count2, probe_spacing);
          if(length(rec_line_idx - vec2(line_idx)) < 0.01f)
          {
            mult = vec4(0.0f, 1.0f, 0.0f, 1.0f);
          }else
          {
            mult = vec4(1.0f, 0.0f, 0.0f, 1.0f);
          }
          mult *= weight;
          color += vec4(1.0f, 0.5f, 0.0f, 1.0f) * mult * 5e-2 * (pixel_t.y - pixel_t.x);
        }*/

        //vec2 t;
        //bool aabb_hit = RayAABBIntersect(pixel_aabb, midpoint0, normalize(midpoint1 - midpoint0), t.x, t.y);
        //if(aabb_hit)
        {
          color += texelFetch(cascade_atlas, atlas_texel_idx, 0) * 0.1f;// * (t.y - t.x);
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


[include: "pcg", "block_probe_layout2", "hrc_basis"]
void FinalGatheringShader2(
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

  color = vec4(0.0f);
  uvec2 line_idx;
  for(line_idx.x = 0u; line_idx.x < block_lines_count2; line_idx.x++)
  {
    uint count = 8u;
    for(uint subpixel_idx = 0u; subpixel_idx < count; subpixel_idx++)
    {
      float ratio = (float(subpixel_idx) + 0.5f) / float(count);
      vec2 line_idxf;
      line_idxf.x = float(line_idx.x) - 0.5f + ratio;
      line_idxf.y = FindLineIdxY(block_idx, line_idxf.x, pixel_pos, block_lines_count2, probe_spacing);

      line_idx.y = uint(round(line_idxf.y));
      int t = int(round(line_idxf.y));
      if(t >= 0 && t < int(block_lines_count2))
      {
        //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
        //color += vec4(0.1f);
        {
          //ivec2 line_idx = (ivec2(round(src_probe_hit.line_idxf)) + ivec2(block_lines_count2)) % ivec2(block_lines_count2);
          //float line_idx_yf = float(line_idx.y);
          Ray discrete_ray = LineIdxfToRay(block_idx, vec2(line_idx.x, line_idx.y), block_lines_count2, probe_spacing);
          Ray continuous_ray = LineIdxfToRay(block_idx, line_idxf, block_lines_count2, probe_spacing);

          ivec2 atlas_texel_idx = IntervalIdxToAtlasTexelIdx(block_idx, ivec2(line_idx), block_lines_count2);

          ivec2 test_probe_idx = ivec2(3, 7);
          vec4 test_aabb = vec4(test_probe_idx, test_probe_idx + ivec2(1)) * float(c0_probe_spacing);

          //float weight = GetLineIdxfWeight(block_idx, vec2(float(line_idx.x), line_idx_yf), block_lines_count2, probe_spacing);
          
          /*vec2 pixel_t;
          bool pixel_aabb_hit = RayAABBIntersect(pixel_aabb, discrete_ray.origin, normalize(discrete_ray.dir), pixel_t.x, pixel_t.y);
          vec2 light_t;
          bool light_aabb_hit = RayAABBIntersect(test_aabb, discrete_ray.origin, normalize(discrete_ray.dir), light_t.x, light_t.y);
          if(pixel_aabb_hit && light_aabb_hit)
          {
            vec4 mult = vec4(1.0f);
            vec2 rec_line_idx = RayToLineIdxf(block_idx, discrete_ray.origin, discrete_ray.dir, block_lines_count2, probe_spacing);
            if(length(rec_line_idx - vec2(line_idx)) < 0.01f)
            {
              mult = vec4(0.0f, 1.0f, 0.0f, 1.0f);
            }else
            {
              mult = vec4(1.0f, 0.0f, 0.0f, 1.0f);
            }
            mult *= weight;
            color += vec4(1.0f, 0.5f, 0.0f, 1.0f) * mult * 5e-2 * (pixel_t.y - pixel_t.x);
          }*/

          //vec2 t;
          //bool aabb_hit = RayAABBIntersect(pixel_aabb, midpoint0, normalize(midpoint1 - midpoint0), t.x, t.y);
          //if(aabb_hit)
          {
            color += texelFetch(cascade_atlas, atlas_texel_idx, 0) * 0.1f / float(count);// * (t.y - t.x);
          }
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

  if((interval_idx.block_idx.x == 8) && (interval_idx.block_idx.y == 4))
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

  uint c0_probe_spacing = 30;
  uint c0_probe_points_count = 20;

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

  /*SetCascade(
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
  }*/

  /*FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    0,
    extended_cascades[0],
    GetSwapchainImage()
  );*/
  int gather_cascade_idx = SliderInt("Gather cascade_idx", 0, cascades_count - 1, 8);
  IntegerTestShader(GetSwapchainImage());

  /*FinalGatheringShader(
    c0_probe_spacing,
    c0_probe_points_count,
    gather_cascade_idx,
    extended_cascades[gather_cascade_idx],
    GetSwapchainImage()
  );*/
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

