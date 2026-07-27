/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NPPDX_DETAIL_DECL_HPP
#define NPPDX_DETAIL_DECL_HPP

// This file defines abbreviations of common declaration modifier sequences with
// the intent of encouraging their consistent use. Especially when prototyping,
// using the full sequences can be seen as cumbersome and is sometimes avoided,
// which can lead to subtle bugs.

// The abbreviations are of the form NPPDX_DECL_*, where the last part is a
// sequence of one- or two-letter mnemonics for the modifiers. Any of them can
// be omitted except that at least one mnemonic must be present. Some mnemonics
// are mutually exclusive and their relative order is fixed. The
// mutually-exclusive mnemonic groups in the prescribed order are:
//
// * N for [[nodiscard]]
// * S for static
// * C for constexpr
// * E for explicit
// * I for inline or F for __forceinline__
// * H for __host__
// * D for __device__

// Rarer modifiers like [[noreturn]] or __noinline__ still need to be specified
// explicitly.

// Note that __host__ has the same meaning as no modifiers from the last two
// groups, but the H-only option is still provided for being able to more
// clearly document that a function is host-only.

// clang-format off

#define NPPDX_DECL_D                                                                        __device__
#define NPPDX_DECL_H                                                               __host__           
#define NPPDX_DECL_HD                                                              __host__ __device__
#define NPPDX_DECL_I                                                        inline                    
#define NPPDX_DECL_ID                                                       inline          __device__
#define NPPDX_DECL_IH                                                       inline __host__           
#define NPPDX_DECL_IHD                                                      inline __host__ __device__
#define NPPDX_DECL_F                                               __forceinline__                    
#define NPPDX_DECL_FD                                              __forceinline__          __device__
#define NPPDX_DECL_FH                                              __forceinline__ __host__           
#define NPPDX_DECL_FHD                                             __forceinline__ __host__ __device__
#define NPPDX_DECL_E                                      explicit                                    
#define NPPDX_DECL_ED                                     explicit                          __device__
#define NPPDX_DECL_EH                                     explicit                 __host__           
#define NPPDX_DECL_EHD                                    explicit                 __host__ __device__
#define NPPDX_DECL_EI                                     explicit          inline                    
#define NPPDX_DECL_EID                                    explicit          inline          __device__
#define NPPDX_DECL_EIH                                    explicit          inline __host__           
#define NPPDX_DECL_EIHD                                   explicit          inline __host__ __device__
#define NPPDX_DECL_EF                                     explicit __forceinline__                    
#define NPPDX_DECL_EFD                                    explicit __forceinline__          __device__
#define NPPDX_DECL_EFH                                    explicit __forceinline__ __host__           
#define NPPDX_DECL_EFHD                                   explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_C                            constexpr                                             
#define NPPDX_DECL_CD                           constexpr                                   __device__
#define NPPDX_DECL_CH                           constexpr                          __host__           
#define NPPDX_DECL_CHD                          constexpr                          __host__ __device__
#define NPPDX_DECL_CI                           constexpr                   inline                    
#define NPPDX_DECL_CID                          constexpr                   inline          __device__
#define NPPDX_DECL_CIH                          constexpr                   inline __host__           
#define NPPDX_DECL_CIHD                         constexpr                   inline __host__ __device__
#define NPPDX_DECL_CF                           constexpr          __forceinline__                    
#define NPPDX_DECL_CFD                          constexpr          __forceinline__          __device__
#define NPPDX_DECL_CFH                          constexpr          __forceinline__ __host__           
#define NPPDX_DECL_CFHD                         constexpr          __forceinline__ __host__ __device__
#define NPPDX_DECL_CE                           constexpr explicit                                    
#define NPPDX_DECL_CED                          constexpr explicit                          __device__
#define NPPDX_DECL_CEH                          constexpr explicit                 __host__           
#define NPPDX_DECL_CEHD                         constexpr explicit                 __host__ __device__
#define NPPDX_DECL_CEI                          constexpr explicit          inline                    
#define NPPDX_DECL_CEID                         constexpr explicit          inline          __device__
#define NPPDX_DECL_CEIH                         constexpr explicit          inline __host__           
#define NPPDX_DECL_CEIHD                        constexpr explicit          inline __host__ __device__
#define NPPDX_DECL_CEF                          constexpr explicit __forceinline__                    
#define NPPDX_DECL_CEFD                         constexpr explicit __forceinline__          __device__
#define NPPDX_DECL_CEFH                         constexpr explicit __forceinline__ __host__           
#define NPPDX_DECL_CEFHD                        constexpr explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_S                     static                                                       
#define NPPDX_DECL_SD                    static                                             __device__
#define NPPDX_DECL_SH                    static                                    __host__           
#define NPPDX_DECL_SHD                   static                                    __host__ __device__
#define NPPDX_DECL_SI                    static                             inline                    
#define NPPDX_DECL_SID                   static                             inline          __device__
#define NPPDX_DECL_SIH                   static                             inline __host__           
#define NPPDX_DECL_SIHD                  static                             inline __host__ __device__
#define NPPDX_DECL_SF                    static                    __forceinline__                    
#define NPPDX_DECL_SFD                   static                    __forceinline__          __device__
#define NPPDX_DECL_SFH                   static                    __forceinline__ __host__           
#define NPPDX_DECL_SFHD                  static                    __forceinline__ __host__ __device__
#define NPPDX_DECL_SE                    static           explicit                                    
#define NPPDX_DECL_SED                   static           explicit                          __device__
#define NPPDX_DECL_SEH                   static           explicit                 __host__           
#define NPPDX_DECL_SEHD                  static           explicit                 __host__ __device__
#define NPPDX_DECL_SEI                   static           explicit          inline                    
#define NPPDX_DECL_SEID                  static           explicit          inline          __device__
#define NPPDX_DECL_SEIH                  static           explicit          inline __host__           
#define NPPDX_DECL_SEIHD                 static           explicit          inline __host__ __device__
#define NPPDX_DECL_SEF                   static           explicit __forceinline__                    
#define NPPDX_DECL_SEFD                  static           explicit __forceinline__          __device__
#define NPPDX_DECL_SEFH                  static           explicit __forceinline__ __host__           
#define NPPDX_DECL_SEFHD                 static           explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_SC                    static constexpr                                             
#define NPPDX_DECL_SCD                   static constexpr                                   __device__
#define NPPDX_DECL_SCH                   static constexpr                          __host__           
#define NPPDX_DECL_SCHD                  static constexpr                          __host__ __device__
#define NPPDX_DECL_SCI                   static constexpr                   inline                    
#define NPPDX_DECL_SCID                  static constexpr                   inline          __device__
#define NPPDX_DECL_SCIH                  static constexpr                   inline __host__           
#define NPPDX_DECL_SCIHD                 static constexpr                   inline __host__ __device__
#define NPPDX_DECL_SCF                   static constexpr          __forceinline__                    
#define NPPDX_DECL_SCFD                  static constexpr          __forceinline__          __device__
#define NPPDX_DECL_SCFH                  static constexpr          __forceinline__ __host__           
#define NPPDX_DECL_SCFHD                 static constexpr          __forceinline__ __host__ __device__
#define NPPDX_DECL_SCE                   static constexpr explicit                                    
#define NPPDX_DECL_SCED                  static constexpr explicit                          __device__
#define NPPDX_DECL_SCEH                  static constexpr explicit                 __host__           
#define NPPDX_DECL_SCEHD                 static constexpr explicit                 __host__ __device__
#define NPPDX_DECL_SCEI                  static constexpr explicit          inline                    
#define NPPDX_DECL_SCEID                 static constexpr explicit          inline          __device__
#define NPPDX_DECL_SCEIH                 static constexpr explicit          inline __host__           
#define NPPDX_DECL_SCEIHD                static constexpr explicit          inline __host__ __device__
#define NPPDX_DECL_SCEF                  static constexpr explicit __forceinline__                    
#define NPPDX_DECL_SCEFD                 static constexpr explicit __forceinline__          __device__
#define NPPDX_DECL_SCEFH                 static constexpr explicit __forceinline__ __host__           
#define NPPDX_DECL_SCEFHD                static constexpr explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_N       [[nodiscard]]                                                              
#define NPPDX_DECL_ND      [[nodiscard]]                                                    __device__
#define NPPDX_DECL_NH      [[nodiscard]]                                           __host__           
#define NPPDX_DECL_NHD     [[nodiscard]]                                           __host__ __device__
#define NPPDX_DECL_NI      [[nodiscard]]                                    inline                    
#define NPPDX_DECL_NID     [[nodiscard]]                                    inline          __device__
#define NPPDX_DECL_NIH     [[nodiscard]]                                    inline __host__           
#define NPPDX_DECL_NIHD    [[nodiscard]]                                    inline __host__ __device__
#define NPPDX_DECL_NF      [[nodiscard]]                           __forceinline__                    
#define NPPDX_DECL_NFD     [[nodiscard]]                           __forceinline__          __device__
#define NPPDX_DECL_NFH     [[nodiscard]]                           __forceinline__ __host__           
#define NPPDX_DECL_NFHD    [[nodiscard]]                           __forceinline__ __host__ __device__
#define NPPDX_DECL_NE      [[nodiscard]]                  explicit                                    
#define NPPDX_DECL_NED     [[nodiscard]]                  explicit                          __device__
#define NPPDX_DECL_NEH     [[nodiscard]]                  explicit                 __host__           
#define NPPDX_DECL_NEHD    [[nodiscard]]                  explicit                 __host__ __device__
#define NPPDX_DECL_NEI     [[nodiscard]]                  explicit          inline                    
#define NPPDX_DECL_NEID    [[nodiscard]]                  explicit          inline          __device__
#define NPPDX_DECL_NEIH    [[nodiscard]]                  explicit          inline __host__           
#define NPPDX_DECL_NEIHD   [[nodiscard]]                  explicit          inline __host__ __device__
#define NPPDX_DECL_NEF     [[nodiscard]]                  explicit __forceinline__                    
#define NPPDX_DECL_NEFD    [[nodiscard]]                  explicit __forceinline__          __device__
#define NPPDX_DECL_NEFH    [[nodiscard]]                  explicit __forceinline__ __host__           
#define NPPDX_DECL_NEFHD   [[nodiscard]]                  explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_NC      [[nodiscard]]        constexpr                                             
#define NPPDX_DECL_NCD     [[nodiscard]]        constexpr                                   __device__
#define NPPDX_DECL_NCH     [[nodiscard]]        constexpr                          __host__           
#define NPPDX_DECL_NCHD    [[nodiscard]]        constexpr                          __host__ __device__
#define NPPDX_DECL_NCI     [[nodiscard]]        constexpr                   inline                    
#define NPPDX_DECL_NCID    [[nodiscard]]        constexpr                   inline          __device__
#define NPPDX_DECL_NCIH    [[nodiscard]]        constexpr                   inline __host__           
#define NPPDX_DECL_NCIHD   [[nodiscard]]        constexpr                   inline __host__ __device__
#define NPPDX_DECL_NCF     [[nodiscard]]        constexpr          __forceinline__                    
#define NPPDX_DECL_NCFD    [[nodiscard]]        constexpr          __forceinline__          __device__
#define NPPDX_DECL_NCFH    [[nodiscard]]        constexpr          __forceinline__ __host__           
#define NPPDX_DECL_NCFHD   [[nodiscard]]        constexpr          __forceinline__ __host__ __device__
#define NPPDX_DECL_NCE     [[nodiscard]]        constexpr explicit                                    
#define NPPDX_DECL_NCED    [[nodiscard]]        constexpr explicit                          __device__
#define NPPDX_DECL_NCEH    [[nodiscard]]        constexpr explicit                 __host__           
#define NPPDX_DECL_NCEHD   [[nodiscard]]        constexpr explicit                 __host__ __device__
#define NPPDX_DECL_NCEI    [[nodiscard]]        constexpr explicit          inline                    
#define NPPDX_DECL_NCEID   [[nodiscard]]        constexpr explicit          inline          __device__
#define NPPDX_DECL_NCEIH   [[nodiscard]]        constexpr explicit          inline __host__           
#define NPPDX_DECL_NCEIHD  [[nodiscard]]        constexpr explicit          inline __host__ __device__
#define NPPDX_DECL_NCEF    [[nodiscard]]        constexpr explicit __forceinline__                    
#define NPPDX_DECL_NCEFD   [[nodiscard]]        constexpr explicit __forceinline__          __device__
#define NPPDX_DECL_NCEFH   [[nodiscard]]        constexpr explicit __forceinline__ __host__           
#define NPPDX_DECL_NCEFHD  [[nodiscard]]        constexpr explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_NS      [[nodiscard]] static                                                       
#define NPPDX_DECL_NSD     [[nodiscard]] static                                             __device__
#define NPPDX_DECL_NSH     [[nodiscard]] static                                    __host__           
#define NPPDX_DECL_NSHD    [[nodiscard]] static                                    __host__ __device__
#define NPPDX_DECL_NSI     [[nodiscard]] static                             inline                    
#define NPPDX_DECL_NSID    [[nodiscard]] static                             inline          __device__
#define NPPDX_DECL_NSIH    [[nodiscard]] static                             inline __host__           
#define NPPDX_DECL_NSIHD   [[nodiscard]] static                             inline __host__ __device__
#define NPPDX_DECL_NSF     [[nodiscard]] static                    __forceinline__                    
#define NPPDX_DECL_NSFD    [[nodiscard]] static                    __forceinline__          __device__
#define NPPDX_DECL_NSFH    [[nodiscard]] static                    __forceinline__ __host__           
#define NPPDX_DECL_NSFHD   [[nodiscard]] static                    __forceinline__ __host__ __device__
#define NPPDX_DECL_NSE     [[nodiscard]] static           explicit                                    
#define NPPDX_DECL_NSED    [[nodiscard]] static           explicit                          __device__
#define NPPDX_DECL_NSEH    [[nodiscard]] static           explicit                 __host__           
#define NPPDX_DECL_NSEHD   [[nodiscard]] static           explicit                 __host__ __device__
#define NPPDX_DECL_NSEI    [[nodiscard]] static           explicit          inline                    
#define NPPDX_DECL_NSEID   [[nodiscard]] static           explicit          inline          __device__
#define NPPDX_DECL_NSEIH   [[nodiscard]] static           explicit          inline __host__           
#define NPPDX_DECL_NSEIHD  [[nodiscard]] static           explicit          inline __host__ __device__
#define NPPDX_DECL_NSEF    [[nodiscard]] static           explicit __forceinline__                    
#define NPPDX_DECL_NSEFD   [[nodiscard]] static           explicit __forceinline__          __device__
#define NPPDX_DECL_NSEFH   [[nodiscard]] static           explicit __forceinline__ __host__           
#define NPPDX_DECL_NSEFHD  [[nodiscard]] static           explicit __forceinline__ __host__ __device__
#define NPPDX_DECL_NSC     [[nodiscard]] static constexpr                                             
#define NPPDX_DECL_NSCD    [[nodiscard]] static constexpr                                   __device__
#define NPPDX_DECL_NSCH    [[nodiscard]] static constexpr                          __host__           
#define NPPDX_DECL_NSCHD   [[nodiscard]] static constexpr                          __host__ __device__
#define NPPDX_DECL_NSCI    [[nodiscard]] static constexpr                   inline                    
#define NPPDX_DECL_NSCID   [[nodiscard]] static constexpr                   inline          __device__
#define NPPDX_DECL_NSCIH   [[nodiscard]] static constexpr                   inline __host__           
#define NPPDX_DECL_NSCIHD  [[nodiscard]] static constexpr                   inline __host__ __device__
#define NPPDX_DECL_NSCF    [[nodiscard]] static constexpr          __forceinline__                    
#define NPPDX_DECL_NSCFD   [[nodiscard]] static constexpr          __forceinline__          __device__
#define NPPDX_DECL_NSCFH   [[nodiscard]] static constexpr          __forceinline__ __host__           
#define NPPDX_DECL_NSCFHD  [[nodiscard]] static constexpr          __forceinline__ __host__ __device__
#define NPPDX_DECL_NSCE    [[nodiscard]] static constexpr explicit                                    
#define NPPDX_DECL_NSCED   [[nodiscard]] static constexpr explicit                          __device__
#define NPPDX_DECL_NSCEH   [[nodiscard]] static constexpr explicit                 __host__           
#define NPPDX_DECL_NSCEHD  [[nodiscard]] static constexpr explicit                 __host__ __device__
#define NPPDX_DECL_NSCEI   [[nodiscard]] static constexpr explicit          inline                    
#define NPPDX_DECL_NSCEID  [[nodiscard]] static constexpr explicit          inline          __device__
#define NPPDX_DECL_NSCEIH  [[nodiscard]] static constexpr explicit          inline __host__           
#define NPPDX_DECL_NSCEIHD [[nodiscard]] static constexpr explicit          inline __host__ __device__
#define NPPDX_DECL_NSCEF   [[nodiscard]] static constexpr explicit __forceinline__                    
#define NPPDX_DECL_NSCEFD  [[nodiscard]] static constexpr explicit __forceinline__          __device__
#define NPPDX_DECL_NSCEFH  [[nodiscard]] static constexpr explicit __forceinline__ __host__           
#define NPPDX_DECL_NSCEFHD [[nodiscard]] static constexpr explicit __forceinline__ __host__ __device__

// clang-format on

#endif // NPPDX_DETAIL_DECL_HPP
