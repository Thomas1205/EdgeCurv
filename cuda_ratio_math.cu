/* -*-c++-*- */

inline __device__ bool mul_is_negative(const int in1, const int in2) {

  if (in1 == 0 || in2 == 0)
    return false;

  int res_signed = (in1 < 0) ? 1 : 0;
  res_signed += (in2 < 0) ? 1 : 0;
  return (res_signed == 1);
}

//the output variables are marked as unsigned but are calculated as signed
inline __device__ int2 mul32(const int in1, const int in2) { //, int& high_out, int& low_out) {

  uint2 out; //.x represents lower bits, .y represents higher bits

  int res_signed = (in1 < 0) ? 1 : 0;
  res_signed += (in2 < 0) ? 1 : 0;
  res_signed = res_signed % 2;

  uint ain1 = (uint) abs(in1);
  uint ain2 = (uint) abs(in2);

  uint h_in1 = ain1 >> 16;
  uint l_in1 = ain1 & 0xFFFF;
  uint h_in2 = ain2 >> 16;
  uint l_in2 = ain2 & 0xFFFF;

  out.x = l_in1*l_in2;
  out.y = h_in1*h_in2;

  uint inter1 = h_in1*l_in2;
  uint inter2 = l_in1*h_in2;

  out.y += inter1 >> 16;
  inter1 = inter1 << 16;
  out.y += inter2 >> 16;
  inter2 = inter2 << 16;

  uint inter3 = inter1 + inter2;
    
  //check for overflow
  if (inter3 < max(inter1, inter2)) 
    out.y++;

  uint inter4 = out.x + inter3;

  if (inter4 < max((uint) out.x,inter3)) 
    out.y++;

  out.x = inter4;

  if (res_signed) {
    out.y = ~out.y;
    out.x = ~out.x;
    out.x += 1;
    if (out.x == 0)
      //add carry flag
      out.y += 1;
  }

  int2 real_out;
  real_out.x = (int) out.x;
  real_out.y = (int) out.y;

  return real_out;
}


//same as above, but for unsigned ints
inline __device__ uint2 umul32(const uint in1, const uint in2) { 

  uint2 out; //.x represents lower bits, .y represents higher bits

  uint h_in1 = in1 >> 16;
  uint l_in1 = in1 & 0xFFFF;
  uint h_in2 = in2 >> 16;
  uint l_in2 = in2 & 0xFFFF;

  out.x = l_in1*l_in2;
  out.y = h_in1*h_in2;

  uint inter1 = h_in1*l_in2;
  uint inter2 = l_in1*h_in2;

  out.y += inter1 >> 16;
  inter1 = inter1 << 16;
  out.y += inter2 >> 16;
  inter2 = inter2 << 16;

  uint inter3 = inter1 + inter2;

  //check for overflow
  if (inter3 < max(inter1, inter2)) 
    out.y++;

  uint inter4 = out.x + inter3;

  if (inter4 < max(out.x,inter3)) 
    out.y++;

  out.x = inter4;

  return out;
}

inline __device__ uint div64(const uint high_in, const uint low_in, const uint divident, int* status) {

  uint2 temp;

  uint high_div = high_in / divident;
  uint high_mod = high_in % divident;

  if (high_div != 0) 
    status[1] = 1;

  uint low_div = low_in / divident;
  uint low_mod = low_in % divident;

  //Note: 0xFFFFFFFF is 2**32 -1, but we want 2**32 itself!
  uint shift_div = 0xFFFFFFFF / divident; 
  uint shift_mod = 0xFFFFFFFF % divident;

  //add the missing one
  shift_mod++;
  if (shift_mod == divident) {
    shift_div++;
    shift_mod = 0;
  }

  uint result = low_div;

  temp = umul32(high_mod,shift_div);

  if (temp.y != 0) {
    status[1] = 2;
  }
  result += temp.x;

  temp = umul32(high_mod,shift_mod);
    
  if (temp.y != 0) {
    status[1] = 4;
  }
  result += temp.x / divident;

  temp.x = temp.x % divident;

  if ((temp.x + low_mod) >= 2*divident) {
    status[1] = 8;
  }
  if ((temp.x + low_mod) >= divident)
    result++;

  return result;
}

#if 0
inline __device__ bool dist_lower(const int old_num, const int old_denom, const int new_num, const int new_denom, 
				  const int ratio_num, const int ratio_denom, int* status) {

  int num_diff = old_num - new_num;
  int denom_diff = old_denom - new_denom;

  int2 check = mul32(denom_diff,ratio_num);

  uint unsigned_hi = (uint) check.y;
  uint unsigned_lo = (uint) check.x;

  if ((unsigned_hi & 0x80000000) != 0) {
    //was signed...
    unsigned_hi = ~unsigned_hi;
    unsigned_lo = ~unsigned_lo;
    unsigned_lo += 1;
    if (unsigned_lo == 0xFFFFFFFF)
      unsigned_hi += 1;
  }

  uint uratio_denom = (uint) ratio_denom;
  int div = div64(unsigned_hi,unsigned_lo,uratio_denom,status);

  if ((div & 0x80000000) != 0) {
    status[1] = 16;
  }
  if ((check.y & 0x80000000) != 0) 
    div *= -1;

  return (num_diff > div);
}

#else

inline __device__ bool dist_lower(const int old_num, const int old_denom, const int new_num, const int new_denom, 
				  const int ratio_num, const int ratio_denom, int* status) {

  int num_diff = old_num - new_num;
  int denom_diff = old_denom - new_denom;

  //NOTE: ratio_denom will always be non-negative

  bool p1_negative = mul_is_negative(denom_diff,ratio_num);
  bool p2_negative = (num_diff < 0);  //mul_is_negative(num_diff,ratio_denom);

  if (p1_negative && !p2_negative)
    return true;
  else if (!p1_negative && p2_negative) 
    return false;
  else {

    uint2 prod1 = umul32(abs(denom_diff),abs(ratio_num));
    uint2 prod2 = umul32(abs(num_diff),ratio_denom);

    if (!p1_negative) {
      if (prod1.y == prod2.y)
	return (prod2.x > prod1.x);
      else
	return (prod2.y > prod1.y);
    }
    else {
      if (prod1.y == prod2.y)
	return (prod2.x < prod1.x);
      else
	return (prod2.y < prod1.y);
    }
  }
}

#endif
